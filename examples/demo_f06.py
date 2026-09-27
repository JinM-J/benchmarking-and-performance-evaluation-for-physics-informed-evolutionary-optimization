#!/usr/bin/env python3
"""Small F6 demonstration: construct an analytic field, train GP, and run DE.

The demonstration uses 1024x401 stored points, population 12, 4 generations
and 12 HF state queries. The paper uses a 24553x9601 field and an F6 budget
of 70 HF queries; this smaller example checks the end-to-end workflow.
Existing benchmark data and results are protected from overwrite.
All child commands use the current interpreter, sys.executable.
"""
from pathlib import Path
import argparse
import json
import os
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--out", type=Path, default=Path("demo_output/f6_gp_seed450"))
    parser.add_argument("--protocol", default="demo_f06", help="Protocol name; alternatively an absolute YAML path for validation")
    parser.add_argument("--seed", type=int, default=450)
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.out.resolve()
    protocol = Path(args.protocol)
    if protocol.is_absolute():
        if protocol.suffix != ".yaml":
            parser.error("An absolute --protocol path must end in .yaml")
        protocol_file = protocol
        protocol_argument = str(protocol.with_suffix(""))
    else:
        protocol_file = root / "protocols" / (args.protocol + ".yaml")
        protocol_argument = args.protocol
    required = [root / "dataset/generate_f06.py", root / "experiments/run.py", protocol_file]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing demo directory: {output}. Choose a new --out.")
    output.mkdir(parents=True, exist_ok=False)
    notice = ("Small-scale F6 workflow demonstration.\n"
              "Coarse manufactured F6 field: refinement=1, 1024x401.\n"
              "GP-DE: seed=%d, population=12, generations=4, HF budget=12 (6+2x3).\n"
              "Paper settings: 24553x9601 field grid and F6 HF budget 70.\n" % args.seed)
    (output / "DEMO_ONLY.txt").write_text(notice, encoding="utf-8")
    print(notice, flush=True)
    dataset = output / "demo_f06_coarse.npz"
    run_dir = output / "run"
    commands = [
        [sys.executable, str(root / "dataset/generate_f06.py"), "--refinement", "1", "--out", str(dataset)],
        [sys.executable, str(root / "experiments/run.py"), "--problem", "f06", "--method", "gp_de",
         "--protocol", protocol_argument, "--seed", str(args.seed), "--data", str(dataset), "--out", str(run_dir)],
    ]
    environment = dict(os.environ)
    environment.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
    (output / "commands.json").write_text(json.dumps(dict(executable=sys.executable, commands=commands), indent=2)+"\n", encoding="utf-8")
    for index, command in enumerate(commands):
        completed = subprocess.run(command, cwd=root, env=environment, text=True,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        (output / f"step_{index+1}.log").write_text(completed.stdout, encoding="utf-8")
        print(completed.stdout, end="", flush=True)
        completed.check_returncode()
    import numpy as np
    with np.load(run_dir / "history.npz", allow_pickle=False) as history:
        manifest = json.loads(str(history["manifest_json"]))
        archive_count = int(history["archive_real_obj"].size)
        if not np.isfinite(history["archive_real_obj"]).all():
            raise AssertionError("Demo archive contains nonfinite objective values")
    if manifest["budget"]["n_state_queries"] != 12 or archive_count != 12:
        raise AssertionError("Expected exactly 12 consumed HF state queries and archive points")
    if manifest["budget"]["surrogate_train_calls"] != 3:
        raise AssertionError("Expected initial GP fit and two update fits")
    summary = dict(status="PASS", purpose="installation_demo_not_paper_reproduction", seed=args.seed,
                   field_grid=[1024,401], dataset_bytes=dataset.stat().st_size,
                   budget=manifest["budget"], archive_size=archive_count,
                   result_selection=manifest["metrics"]["result_selection"],
                   best_real_objective_archive=manifest["metrics"]["best_real_objective_archive"],
                   history="run/history.npz", resolved_config="run/resolved_config.yaml")
    (output / "demo_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
