"""Check the F3 reference candidate by substitution and optional PDE-field evaluation.

The algebraic zero target alone does not establish PDE reachability."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from reference.common.candidates import verify


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path,
                        help="Also evaluate the recorded decision on f03.npz in this directory")
    parser.add_argument("--output", type=Path,
                        help="Write a new JSON file; otherwise print JSON to stdout")
    args = parser.parse_args()
    if args.output and args.output.exists():
        parser.error("Output exists; choose a different file.")
    result = verify("f03", args.data_dir)
    payload = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    else:
        print(payload, end="")
    return 0 if result["requested_checks_complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
