"""Portable sequential launcher; --dry-run prints the plan without running experiments."""
import argparse
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core.registry import DATA_FILES, METHODS, load_protocol

MAIN_PROBLEMS = [f"f{i:02d}" for i in range(1, 12)]
MAIN_METHODS = ["gp_de", "pigp_de", "pinn_de", "rbfn_de", "mlp_de",
                "pino_de", "ji_sade_grm", "glosade"]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--protocol", default="main")
    ap.add_argument("--problems", default=",".join(MAIN_PROBLEMS))
    ap.add_argument("--methods", default=",".join(MAIN_METHODS))
    ap.add_argument("--base-seed", type=int, default=450)
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--output-root", type=Path, default=ROOT / "results")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--cpu", action="store_true", help="Hide CUDA devices from each child process")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if args.runs < 1 or args.threads < 1:
        ap.error("--runs and --threads must be positive")
    problems = [s.strip() for s in args.problems.split(",") if s.strip()]
    methods = [s.strip() for s in args.methods.split(",") if s.strip()]
    if not problems or not methods or len(set(problems)) != len(problems) or len(set(methods)) != len(methods):
        ap.error("Problem/method lists must be nonempty and contain no duplicates")
    if any(p not in DATA_FILES for p in problems) or any(m not in METHODS for m in methods):
        ap.error("Unknown problem or method; see README.md and core/registry.py")
    for p in problems:
        load_protocol(ROOT / "protocols" / f"{args.protocol}.yaml", p.upper())
    jobs = []
    for p in problems:
        for m in methods:
            for seed in range(args.base_seed, args.base_seed + args.runs):
                out = args.output_root.resolve() / p / m / f"seed{seed}"
                cmd = [sys.executable, str(ROOT / "experiments/run.py"), "--problem", p,
                       "--method", m, "--protocol", args.protocol, "--seed", str(seed),
                       "--out", str(out)]
                jobs.append((p, out, cmd))
    print(f"[PLAN] {len(jobs)} runs, sequential execution; protocol={args.protocol}")
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
