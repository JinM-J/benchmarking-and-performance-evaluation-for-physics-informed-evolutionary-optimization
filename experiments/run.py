# experiments/run.py
"""Run a (problem, method, protocol, seed) combination.

From the repository root:
    python experiments/run.py --problem f01 --method gp_de --protocol paper --seed 450

Paper-protocol outputs under results/main/<problem>/<method>/seed<seed>/:
    history.npz: consumed-HF archive, final population, and snapshot history.
    resolved_config.yaml: complete resolved configuration for this run."""
import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.registry import make_problem, make_method, load_protocol, DATA_FILES
from core.experiment import Experiment
from core.version import FRAMEWORK_VERSION, SCHEMA_VERSION, BENCHMARK_VERSION
from evaluation.metrics.constraint import FEASIBLE_TOL
from evaluation.metrics.optimization import best_feasible_objective
from optimizers.de import (EPSILON_THETA_FRAC, EPSILON_CONTROL_FRAC,
                           EPSILON_POWER)

ARCHIVE_BEST_SCALAR_VERSION = "archive_best_scalar_v1"


def main():
    ap = argparse.ArgumentParser(description="Run the PDE-constrained optimization benchmark")
    ap.add_argument("--problem", required=True, help="Problem identifier, e.g. f01")
    ap.add_argument("--method", required=True, help="Method identifier, e.g. gp_de")
    ap.add_argument("--protocol", default="paper", help="Protocol name (protocols/<name>.yaml); default: paper")
    ap.add_argument("--seed", type=int, default=450)
    ap.add_argument("--data", default=None, help="Reference NPZ path (default: dataset/fNN.npz)")
    ap.add_argument("--protocol-key", default=None,
                    help="Problem key in the protocol (default: --problem); external problems may reuse a key such as f01")
    ap.add_argument("--out", default=None, help="Output directory; paper-protocol default: results/main/<problem>/<method>/seed<seed>")
    ap.add_argument("--export-surrogate-grid", type=int, default=0,
                    help="Export the final field or central 3D slice; nodes per axis, 0 disables; no additional fitting or HF queries")
    ap.add_argument("--save-final-surrogate", action="store_true",
                    help="Save final inference state and configuration without additional plots")
    ap.add_argument("--defer-pde-residual", action="store_true",
                    help="With model saving, defer final PDE residual evaluation to an offline derivative evaluation")
    args = ap.parse_args()
    if args.defer_pde_residual and not args.save_final_surrogate:
        ap.error("Deferred residual evaluation requires saving the final surrogate model")

    problem_key = args.protocol_key or args.problem.upper()
    protocol_path = ROOT / "protocols" / f"{args.protocol}.yaml"
    protocol = load_protocol(protocol_path, problem_key)
    protocol_document = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    configuration = {
        "protocol_file": str(protocol_path.relative_to(ROOT)),
        "suite_version": protocol_document.get("suite_version"),
    }

    problem = make_problem(args.problem)
    method = make_method(args.method, protocol, args.seed)

    if args.data:
        data_path = args.data
    elif args.problem in DATA_FILES:
        data_path = str(ROOT / "dataset" / DATA_FILES[args.problem])
    elif hasattr(problem, "dataset_fingerprint"):
        data_path = None   # Self-contained problems use an online deterministic backend.
    else:
        raise KeyError(
            f"External problem {args.problem!r} has no built-in data mapping; provide an NPZ path with --data"
        )
    if data_path is not None and not Path(data_path).is_file():
        raise FileNotFoundError(
            f"Missing reference dataset: {data_path}. "
            "See dataset/README.md; no data are generated automatically."
        )
    results_root = ROOT / "results" / "main" if args.protocol == "paper" else ROOT / "results"
    out_dir = Path(args.out) if args.out else (
        results_root / args.problem / args.method / f"seed{args.seed}"
    )
    if args.save_final_surrogate:
        if getattr(method, "needs_decision_evaluator", False):
            ap.error("Final state-field snapshots require a field-surrogate method")
        if any((out_dir / name).exists() for name in
               ("history.npz", "final_surrogate.pkl", "resolved_config.yaml")):
            ap.error("Model export requires a new output directory; existing models or results cannot be overwritten")
    if args.export_surrogate_grid:
        if args.export_surrogate_grid < 2 or np.asarray(problem.query_bounds).shape not in ((2, 2), (3, 2)):
            ap.error("Final-field export requires a 2D/3D query domain and at least two nodes per axis")
        if (out_dir / "history.npz").exists():
            ap.error("Field illustrations require a new output directory to preserve existing optimization results")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[RUN] problem={args.problem} method={args.method} "
          f"protocol={protocol.name} seed={args.seed}")
    print(f"[RUN] data: {data_path}")
    print(f"[RUN] HF budget: {protocol.hf_budget}")

    exp = Experiment(problem, method, protocol, data_path, args.seed)
    t0 = time.perf_counter()
    res = exp.run(evaluate_final_residual=not args.defer_pde_residual)
    wall_total = time.perf_counter() - t0
    if args.save_final_surrogate:
        # Save immediately after optimization so later export failures retain the final model.
        from experiments.surrogate_checkpoint import save_checkpoint
        save_checkpoint(method.surrogate, out_dir / "final_surrogate.pkl", args.problem,
                        metadata=dict(problem=args.problem, method=args.method, seed=args.seed,
                                      protocol=asdict(protocol), surrogate_config=method.surrogate_config,
                                      data_path=data_path, last_update_generation=int(res.db_history[-1]["gen"]),
                                      hf_queries=int(res.n_state_queries),
                                      train_calls=int(res.surrogate_train_calls)))

    # Persist schema v1.2: official HF archive and final-population diagnostics.
    if data_path is not None:
        dataset_file = Path(data_path).name
    else:
        dataset_file = None
    best_fit = float(np.min(res.fitness)) if len(res.fitness) else float("nan")
    final_mse200 = float(res.db_history[-1]["mse200"]) if res.db_history else float("nan")
    # Official result: best feasible true objective among consumed HF evaluations.
    best_real_obj = best_feasible_objective(
        res.archive_real_obj, res.archive_real_vio
    )
    archive_feas_flags = (
        np.isfinite(res.archive_real_vio)
        & (res.archive_real_vio <= FEASIBLE_TOL)
        & np.isfinite(res.archive_real_obj)
    )
    if archive_feas_flags.any():
        feasible_indices = np.flatnonzero(archive_feas_flags)
        archive_best_idx = int(feasible_indices[
            np.argmin(res.archive_real_obj[archive_feas_flags])
        ])
        archive_best_cv = float(res.archive_real_vio[archive_best_idx])
    else:
        archive_best_idx = None
        finite_vio = res.archive_real_vio[np.isfinite(res.archive_real_vio)]
        archive_best_cv = (float(np.min(finite_vio))
                           if finite_vio.size else float("nan"))
    final_feas_flags = res.final_real_vio <= FEASIBLE_TOL
    final_pop_best = best_feasible_objective(
        res.final_real_obj, res.final_real_vio
    )
    # Optional observable MSE; absent when undefined (NaN).
    vmse_final = float(res.db_history[-1].get("voltage_mse", float("nan"))) \
        if res.db_history else float("nan")

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "framework_version": FRAMEWORK_VERSION,
        "benchmark_version": BENCHMARK_VERSION,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "problem": args.problem,
        "method": args.method,
        "protocol": protocol.name,
        "configuration": configuration,
        "seed": int(args.seed),
        "dataset": {"file": dataset_file},
        "budget": {
            "hf_budget_protocol": int(protocol.hf_budget),
            "budget_kind": getattr(protocol, "budget_kind", "state"),
            "n_state_queries": int(res.n_state_queries),
            "n_parameter_queries": int(res.n_parameter_queries),
            "surrogate_train_calls": int(res.surrogate_train_calls),
            "hf_wall_time_sec": round(res.hf_wall_time, 6),
            "surrogate_train_time_sec": round(res.surrogate_train_time, 6),
            "surrogate_infer_time_sec": round(res.surrogate_infer_time, 6),
        },
        "metrics": {
            "best_fitness_penalized_final_pop": best_fit,
            "mse200_final": final_mse200,
            "n_snapshots": len(res.db_history),
            # Official feasibility tolerance is 1e-4; rank consumed-HF archive results.
            # Final-population diagnostics must not replace archive-based paper rankings.
            "feasible_tol": FEASIBLE_TOL,
            "result_selection": "best_feasible_hf_archive",
            "archive_size": int(res.archive_real_obj.size),
            "archive_has_feasible": bool(archive_feas_flags.any()),
            "archive_best_index": archive_best_idx,
            "archive_best_source": "explicit_hf_archive",
            "archive_best_feasible_tol": FEASIBLE_TOL,
            "archive_best_scalar_version": ARCHIVE_BEST_SCALAR_VERSION,
            "best_real_objective": best_real_obj,
            "best_real_objective_archive": best_real_obj,
            "cv_at_archive_best": archive_best_cv,
            "best_real_objective_final_pop": final_pop_best,
            "feasible_rate_final_pop": float(final_feas_flags.mean()),
            "cv_mean_final_pop": float(np.mean(res.final_real_vio)),
            "cv_max_final_pop": float(np.max(res.final_real_vio)),
            "pde_residual_final": float(res.pde_residual_final),
            # Operator-PIGP linear residual; NaN and excluded from tables for other surrogates.
            "pde_residual_linear_final": float(res.pde_residual_linear_final),
            "voltage_mse_final": vmse_final,
            "wall_time_total_sec": round(wall_total, 6),
        },
    }
    if args.defer_pde_residual:
        manifest["supplement"] = {
            "purpose": "Final-surrogate checkpoint for offline evaluation",
            "included_in_original_main_statistics": False,
            "pde_residual_status": "deferred_to_offline_derivative_audit",
        }
    if method.constraint_mode == "epsilon":
        manifest["constraint_handler"] = {
            "mode": "epsilon",
            "epsilon_0": "initial_violation_descending_order_statistic",
            "theta_fraction": EPSILON_THETA_FRAC,
            "control_generation_fraction": EPSILON_CONTROL_FRAC,
            "power": EPSILON_POWER,
            "parent_trial_same_generation_epsilon": True,
        }

    np.savez_compressed(
        out_dir / "history.npz",
        final_pop=res.pop,
        final_fit=res.fitness,
        db_history=np.array(res.db_history, dtype=object),
        # Store true diagnostics as top-level fields without changing snapshot structure.
        real_watch_gen=np.array(res.real_watch_gen, dtype=np.int64),
        real_obj_best=np.array(res.real_obj_best, dtype=float),
        real_feasible_ratio=np.array(res.real_feasible_ratio, dtype=float),
        # Official candidates are archived decisions evaluated through consumed HF calls.
        archive_dec=res.archive_dec,
        archive_real_obj=res.archive_real_obj,
        archive_real_vio=res.archive_real_vio,
        result_selection="best_feasible_hf_archive",
        # Readable scalar result, consistent with the archive arrays for this run.
        best_real_objective_archive=np.float64(best_real_obj),
        cv_at_archive_best=np.float64(archive_best_cv),
        archive_best_index=np.int64(
            archive_best_idx if archive_best_idx is not None else -1
        ),
        archive_has_feasible=np.bool_(archive_feas_flags.any()),
        archive_best_source="explicit_hf_archive",
        archive_best_feasible_tol=np.float64(FEASIBLE_TOL),
        archive_best_scalar_version=ARCHIVE_BEST_SCALAR_VERSION,
        # Retain the final population for diagnostics only, outside official best-objective ranking.
        final_real_obj=res.final_real_obj,
        final_real_vio=res.final_real_vio,
        seed=np.int64(args.seed),
        schema_version=SCHEMA_VERSION,
        framework_version=FRAMEWORK_VERSION,
        benchmark_version=BENCHMARK_VERSION,
        manifest_json=json.dumps(manifest, ensure_ascii=False),
    )

    resolved = {
        "schema_version": SCHEMA_VERSION,
        "framework_version": FRAMEWORK_VERSION,
        "problem": args.problem,
        "method": args.method,
        "protocol": protocol.name,
        "configuration": configuration,
        "protocol_config": asdict(protocol),
        "seed": args.seed,
        "data_path": str(data_path),
        "protocol_params": {
            "pop_size": protocol.pop_size,
            "maxgen": protocol.maxgen,
            "init_points": protocol.init_points,
            "update_interval": protocol.update_interval,
            "hf_points_per_update": protocol.hf_points_per_update,
            "penalty_min": protocol.penalty_min,
            "penalty_max": protocol.penalty_max,
            "constraint_mode": method.constraint_mode,
            "hf_query_budget": protocol.hf_budget,
            "update_policy": dict(protocol.update_policy),
            "method_params": dict(protocol.method_params),
        },
        "surrogate_config": method.surrogate_config,
        "hf_budget": protocol.hf_budget,
        "n_hf_queries": res.n_hf_queries,
        "n_state_queries": res.n_state_queries,
        "n_parameter_queries": res.n_parameter_queries,
        "hf_wall_time_sec": round(res.hf_wall_time, 6),
        "surrogate_train_calls": res.surrogate_train_calls,
        "result_selection": "best_feasible_hf_archive",
        "archive_size": int(res.archive_real_obj.size),
        "best_real_objective_archive": best_real_obj,
        "cv_at_archive_best": archive_best_cv,
        "archive_best_index": (
            archive_best_idx if archive_best_idx is not None else -1
        ),
        "archive_has_feasible": bool(archive_feas_flags.any()),
        "archive_best_source": "explicit_hf_archive",
        "archive_best_feasible_tol": FEASIBLE_TOL,
        "archive_best_scalar_version": ARCHIVE_BEST_SCALAR_VERSION,
        "best_real_objective_final_pop": final_pop_best,
        "mse200_series": [s["mse200"] for s in res.db_history],
    }
    if method.constraint_mode == "epsilon":
        resolved["constraint_handler"] = dict(manifest["constraint_handler"])
    with open(out_dir / "resolved_config.yaml", "w", encoding="utf-8") as fp:
        yaml.safe_dump(resolved, fp, allow_unicode=True, sort_keys=False)

    if args.save_final_surrogate:
        from experiments.surrogate_checkpoint import write_checkpoint_metadata
        write_checkpoint_metadata(method.surrogate, out_dir, manifest, res)

    if args.export_surrogate_grid:
        from experiments.export_surrogate_field import export_final_field
        export_final_field(problem, method, protocol, data_path, res,
                           out_dir, args.export_surrogate_grid, manifest,
                           preserve_checkpoint=args.save_final_surrogate)

    print(f"\n[DONE] Snapshots: {len(res.db_history)}, HF count: {res.n_hf_queries}")
    print(f"[DONE] Best true objective in the HF archive: {best_real_obj}")
    print(f"[DONE] Output: {out_dir}")


if __name__ == "__main__":
    main()
