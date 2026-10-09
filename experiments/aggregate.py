"""Summarize schema-1.2 HF archives; no reference search or data loading."""
import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation.metrics.constraint import FEASIBLE_TOL


def summarize_run(path):
    # Read only numeric arrays and JSON; db_history pickled objects are not required.
    with np.load(path, allow_pickle=False) as data:
        manifest = json.loads(str(data["manifest_json"]))
        if str(data["result_selection"]) != "best_feasible_hf_archive":
            raise ValueError(f"Unsupported result-selection semantics: {path}")
        obj = np.asarray(data["archive_real_obj"], dtype=float).reshape(-1)
        vio = np.asarray(data["archive_real_vio"], dtype=float).reshape(-1)
    if obj.shape != vio.shape or obj.size == 0:
        raise ValueError(f"Empty or mismatched HF archive: {path}")
    finite = np.isfinite(obj) & np.isfinite(vio)
    feasible = finite & (vio <= FEASIBLE_TOL)
    budget = manifest["budget"]
    return {
        "problem": manifest["problem"], "method": manifest["method"],
        "protocol": manifest["protocol"], "dataset_file": manifest.get("dataset", {}).get("file") or "online",
        "seed": int(manifest["seed"]), "feasible": bool(feasible.any()),
        "best_f": float(obj[feasible].min()) if feasible.any() else float("nan"),
        "hf": budget["n_state_queries"] if budget.get("budget_kind", "state") == "state"
              else budget["n_parameter_queries"],
        "runtime": manifest["metrics"]["wall_time_total_sec"],
        "train_t": budget["surrogate_train_time_sec"],
        "infer_t": budget["surrogate_infer_time_sec"],
        "mse": manifest["metrics"]["mse200_final"],
    }


def aggregate_rows(rows):
    groups = {}
    seen = set()
    for row in rows:
        key = tuple(row[k] for k in ("problem", "method", "protocol", "dataset_file"))
        run_key = key + (row["seed"],)
        if run_key in seen:
            raise ValueError(f"Duplicate run: {run_key}")
        seen.add(run_key)
        groups.setdefault(key, []).append(row)
    output = []
    for key, runs in sorted(groups.items()):
        result = dict(zip(("problem", "method", "protocol", "dataset_file"), key))
        result.update(n=len(runs), feasible_runs=sum(r["feasible"] for r in runs),
                      feasible_rate=sum(r["feasible"] for r in runs) / len(runs))
        for metric in ("best_f", "hf", "runtime", "train_t", "infer_t", "mse"):
            values = np.array([r[metric] for r in runs], dtype=float)
            values = values[np.isfinite(values)]
            result[metric + "_n"] = len(values)
            result[metric + "_mean"] = float(values.mean()) if len(values) else float("nan")
            result[metric + "_std_ddof0"] = float(values.std(ddof=0)) if len(values) else float("nan")
        output.append(result)
    return output


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", type=Path, help="Results root; paper-protocol default: results/main")
    ap.add_argument("--protocol", default="paper", help="Exact protocol name before /problem")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    args.results = args.results or (ROOT / "results" / "main" if args.protocol == "paper" else ROOT / "results")
    rows = []
    for path in sorted(args.results.glob("*/*/seed*/history.npz")):
        row = summarize_run(path)
        if row["protocol"].split("/", 1)[0] == args.protocol:
            rows.append(row)
    if not rows:
        ap.error("No matching schema-1.2 HF archives found; no results were generated")
    output = args.out or args.results / "summary.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    aggregated = aggregate_rows(rows)
    with output.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=aggregated[0].keys())
        writer.writeheader()
        writer.writerows(aggregated)
    print(f"{len(rows)} runs; {len(aggregated)} groups -> {output}")


if __name__ == "__main__":
    main()
