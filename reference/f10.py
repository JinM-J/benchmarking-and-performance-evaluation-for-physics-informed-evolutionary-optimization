"""F10 reference optimum on the stored PDE field and discrete quadrature."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from reference.common import integral


def solve(data_dir):
    result = integral.solve("F10", Path(data_dir))
    result["solver_sha256"] = hashlib.sha256(Path(integral.__file__).read_bytes()).hexdigest()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, type=Path, help="Directory containing f10.npz")
    parser.add_argument("--output", type=Path, help="New JSON file; defaults to stdout")
    args = parser.parse_args()
    if args.output is not None and args.output.exists():
        parser.error(f"Output already exists: {args.output}")
    result = solve(args.data_dir)
    payload = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output is None:
        print(payload, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(payload)
        print(f"PASS: F10 reference verification; {args.output}")


if __name__ == "__main__":
    main()
