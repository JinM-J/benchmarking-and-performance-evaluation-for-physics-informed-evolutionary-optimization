"""Inspect reference array headers; optionally hash existing files. Never generate data."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--problems", default=None, help="Comma-separated stable code identifiers")
    ap.add_argument("--sha256", action="store_true", help="Read the full file to check archived byte identity")
    ap.add_argument("--list", action="store_true", help="Show the expected catalog without requiring files")
    args = ap.parse_args()
    catalog = json.loads((ROOT / "dataset/reference_manifest.json").read_text())
    selected = set(args.problems.split(",")) if args.problems else {r["code_problem"] for r in catalog}
    unknown = selected - {r["code_problem"] for r in catalog}
    if unknown:
        ap.error(f"Unknown identifiers: {sorted(unknown)}")
    failed = False
    for row in catalog:
        if row["code_problem"] not in selected:
            continue
        path = ROOT / "dataset" / row["file"]
        label = f"{row['paper_problem']} ({row['code_problem']}): {row['file']}"
        if args.list:
            print(label, row["u_shape"], row["u_dtype"])
            continue
        if not path.exists():
            print("MISSING", label)
            failed = True
            continue
        with zipfile.ZipFile(path) as archive, archive.open("u.npy") as fp:
            version = np.lib.format.read_magic(fp)
            if version == (1, 0):
                shape, fortran, dtype = np.lib.format.read_array_header_1_0(fp)
            elif version == (2, 0):
                shape, fortran, dtype = np.lib.format.read_array_header_2_0(fp)
            else:
                raise ValueError(f"Unsupported NPY header version {version} in {path}")
        matches = list(shape) == row["u_shape"] and str(dtype) == row["u_dtype"]
        print("HEADER_OK" if matches else "HEADER_MISMATCH", label, shape, dtype)
        failed |= not matches
        if args.sha256:
            h = hashlib.sha256()
            with path.open("rb") as fp:
                for block in iter(lambda: fp.read(8 * 1024 * 1024), b""):
                    h.update(block)
            matches = h.hexdigest() == row["reference_file_sha256"]
            print("HASH_OK" if matches else "HASH_DIFFERS_FROM_ARCHIVED_FILE", h.hexdigest())
            failed |= not matches
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
