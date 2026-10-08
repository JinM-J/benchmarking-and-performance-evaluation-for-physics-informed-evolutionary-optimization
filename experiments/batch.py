"""Portable sequential launcher; --dry-run prints the plan without running experiments."""
import argparse
import os
from pathlib import Path
import shlex
import subprocess
import sys
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core.registry import DATA_FILES, METHODS, load_protocol, make_method

MAIN_PROBLEMS = [f"f{i:02d}" for i in range(1, 12)]
MAIN_METHODS = ["gp_de", "pigp_de", "pinn_de", "rbfn_de", "mlp_de",
                "pino_de", "ji_sade_grm", "glosade"]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--protocol", default="main")
    ap.add_argument("--group", help="Run group: main, optimizers, constraints, residuals or illustrations; default: main for the paper protocol")
    ap.add_argument("--problems")
    ap.add_argument("--methods")
    ap.add_argument("--base-seed", type=int)
    ap.add_argument("--runs", type=int)
    ap.add_argument("--output-root", type=Path)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--cpu", action="store_true", help="Hide CUDA devices from each child process")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    protocol_path = ROOT / "protocols" / f"{args.protocol}.yaml"
    document = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    groups = document.get("run_groups", {})
    if args.group is None and args.protocol == "main":
        args.group = "main"
    if args.group and args.group not in groups:
        ap.error(f"Unknown run group {args.group!r}; available: {list(groups)}")
    group = groups.get(args.group, {})
    args.base_seed = args.base_seed if args.base_seed is not None else group.get("base_seed", 450)
    args.runs = args.runs if args.runs is not None else group.get("runs", 1)
    args.output_root = args.output_root or (ROOT / "results" / args.group if args.group else ROOT / "results")
    if args.runs < 1 or args.threads < 1:
        ap.error("--runs and --threads must be positive")
    problems = ([s.strip() for s in args.problems.split(",") if s.strip()]
                if args.problems is not None else group.get("problems", MAIN_PROBLEMS))
    methods = ([s.strip() for s in args.methods.split(",") if s.strip()]
               if args.methods is not None else group.get("methods", MAIN_METHODS))
    if not problems or not methods or len(set(problems)) != len(problems) or len(set(methods)) != len(methods):
        ap.error("Problem/method lists must be nonempty and contain no duplicates")
    if any(p not in DATA_FILES for p in problems) or any(m not in METHODS for m in methods):
        ap.error("Unknown problem or method; see README.md and core/registry.py")
    if args.group == "constraints" and "f06" in problems:
        ap.error("F6 has no algebraic constraints; the selected constraint comparison applies to F8")
    jobs = []
    for p in problems:
        protocol = load_protocol(protocol_path, p.upper())
        for m in methods:
            # Validate method/profile availability without fitting or loading data.
            make_method(m, protocol, args.base_seed)
            for seed in range(args.base_seed, args.base_seed + args.runs):
                out = args.output_root.resolve() / p / m / f"seed{seed}"
                cmd = [sys.executable, str(ROOT / "experiments/run.py"), "--problem", p,
                       "--method", m, "--protocol", args.protocol, "--seed", str(seed),
                       "--out", str(out)]
                if group.get("save_final_surrogate", False):
                    cmd.append("--save-final-surrogate")
                if group.get("defer_pde_residual", False):
                    cmd.append("--defer-pde-residual")
                if group.get("export_surrogate_grid", 0):
                    cmd += ["--export-surrogate-grid", str(group["export_surrogate_grid"])]
                jobs.append((p, out, cmd))
    print(f"[PLAN] {len(jobs)} runs, sequential execution; protocol={args.protocol}; group={args.group or 'custom'}")
    if args.dry_run:
        for _, _, cmd in jobs:
            print(shlex.join(cmd))
        return
    # Check the entire plan before launching anything. Never silently overwrite a previous run.
    for p, out, _ in jobs:
        data = ROOT / "dataset" / DATA_FILES[p]
        if not data.is_file():
            ap.error(f"Missing {data}; see dataset/README.md")
        if out.exists() and any(out.iterdir()):
            ap.error(f"Output directory is not empty: {out}; select a new --output-root")
    env = os.environ.copy()
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        env[name] = str(args.threads)
    if args.cpu:
        env["CUDA_VISIBLE_DEVICES"] = ""
    for index, (_, out, cmd) in enumerate(jobs, 1):
        out.mkdir(parents=True, exist_ok=True)
        print(f"[{index}/{len(jobs)}] {out.relative_to(args.output_root.resolve())}", flush=True)
        with (out / "run.log").open("w", encoding="utf-8") as log:
            proc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        if proc.returncode:
            raise SystemExit(f"Run failed ({proc.returncode}); see {out / 'run.log'}")
    print(f"[DONE] {len(jobs)} runs")


if __name__ == "__main__":
    main()
