"""Check the F8 reference candidate by substitution and optional PDE-field evaluation.

The value 3/4 is the algebraic minimum with the exact equality u=x^2.
The experiment uses |u-x^2| <= 1e-3; this check does not certify the optimum
within that tolerance band or establish PDE reachability without field data."""
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
                        help="Also evaluate the recorded decision on f08.npz in this directory")
    parser.add_argument("--output", type=Path,
                        help="Write a new JSON file; otherwise print JSON to stdout")
    args = parser.parse_args()
    if args.output and args.output.exists():
        parser.error("Output exists; choose a different file.")
    result = verify("f08", args.data_dir)
    result["constraint_semantics"] = "The reference value 3/4 uses exact u=x^2; the experiment permits |u-x^2| <= 1e-3. No tolerance-band optimum is certified."
    payload = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    else:
        print(payload, end="")
    return 0 if result["requested_checks_complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
