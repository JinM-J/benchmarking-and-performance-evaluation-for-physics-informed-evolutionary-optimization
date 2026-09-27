"""Import exact reference files under the current paper F1--F11 numbering.

Every selected source and existing destination is checked against the distributed
SHA256 and byte count before any filesystem writes. Files are copied by default;
--link creates hard links on the same filesystem. Existing matching files are
kept, and mismatches are never overwritten. Original files are not renamed.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHUNK_BYTES = 8 * 1024 * 1024


def checked_file(path, row):
    """Check byte identity without unpacking or loading the PDE array."""
    if not path.is_file():
        raise FileNotFoundError(f"Required reference file is missing or not a file: {path}")
    expected_size = int(row["reference_file_bytes"])
    if path.stat().st_size != expected_size:
        raise ValueError(f"Reference size mismatch: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_BYTES), b""):
            digest.update(chunk)
    if digest.hexdigest() != row["reference_file_sha256"]:
        raise ValueError(f"Reference SHA256 mismatch: {path}")


def identity(path):
    """Detect ordinary source changes between preflight and transfer."""
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


def safe_filename(name):
    if not isinstance(name, str) or not name or Path(name).name != name or name in (".", ".."):
        raise ValueError(f"Invalid reference filename in manifest: {name!r}")
    return name


def import_data(source_dir, destination_dir, catalog, *, legacy_numbering=False,
                link=False, problems=None):
    """Validate all selected files, then import only absent canonical filenames."""
    source_dir = Path(source_dir).resolve()
    destination_dir = Path(destination_dir).resolve()
    if source_dir == destination_dir:
        raise ValueError("Source and destination directories must differ; in-place renumbering is not supported")
    if not source_dir.is_dir():
        raise NotADirectoryError(f"Source directory does not exist: {source_dir}")
    if destination_dir.exists() and not destination_dir.is_dir():
        raise NotADirectoryError(f"Destination is not a directory: {destination_dir}")
    known = {row["paper_problem"] for row in catalog}
    selected = known if problems is None else set(problems)
    if not selected or selected - known:
        raise ValueError(f"Unknown or empty problem selection: {sorted(selected - known)}")
    rows = [row for row in catalog if row["paper_problem"] in selected]
    if len({row["file"] for row in rows}) != len(rows):
        raise ValueError("Duplicate destination filenames in reference manifest")
    plan = []
    for row in rows:
        source_name = safe_filename(row["legacy_file"] if legacy_numbering else row["file"])
        target_name = safe_filename(row["file"])
        source = source_dir / source_name
        target = destination_dir / target_name
        print(f"Checking {row['paper_problem']}: {source_name} -> {target_name}", flush=True)
        source_identity = identity(source)
        checked_file(source, row)
        if identity(source) != source_identity:
            raise RuntimeError(f"Source changed during verification: {source}")
        present = target.exists() or target.is_symlink()
        if present:
            checked_file(target, row)
        plan.append((row, source, target, source_identity, present))

    # All source and existing-target checks precede directory creation or transfer.
    destination_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for row, source, target, source_identity, present in plan:
        if identity(source) != source_identity:
            raise RuntimeError(f"Source changed after verification: {source}")
        if present:
            results.append({"problem": row["paper_problem"], "file": target.name,
                            "status": "kept_matching"})
            continue
        if link:
            # os.link is exclusive: a newly appearing destination is never replaced.
            os.link(source, target)
            status = "hard_linked"
        else:
            # Exclusive creation prevents overwrite even if a destination appears
            # after preflight. Hash again while streaming; do not load the NPZ.
            created = False
            try:
                with target.open("xb") as output:
                    created = True
                    digest = hashlib.sha256()
                    copied = 0
                    with source.open("rb") as handle:
                        for chunk in iter(lambda: handle.read(CHUNK_BYTES), b""):
                            output.write(chunk)
                            digest.update(chunk)
                            copied += len(chunk)
                if copied != row["reference_file_bytes"] or digest.hexdigest() != row["reference_file_sha256"]:
                    raise RuntimeError(f"Source bytes changed during copy: {source}")
                status = "copied"
            except BaseException:
                if created:
                    target.unlink(missing_ok=True)
                raise
        print(f"{status}: {target.name}", flush=True)
        results.append({"problem": row["paper_problem"], "file": target.name,
                        "status": status})
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--destination-dir", type=Path, default=ROOT / "dataset")
    parser.add_argument("--legacy-numbering", action="store_true",
                        help="Read original archive names: old 12 becomes paper 6; old 6--10 become paper 7--11")
    parser.add_argument("--link", action="store_true",
                        help="Create hard links instead of copies; source and destination must share a filesystem")
    parser.add_argument("--problems", nargs="+", choices=[f"F{i}" for i in range(1, 12)],
                        help="Import selected paper problems, for example F6 F7 (default: all eleven)")
    args = parser.parse_args()
    catalog = json.loads((ROOT / "dataset/reference_manifest.json").read_text(encoding="utf-8"))
    results = import_data(args.source_dir, args.destination_dir, catalog,
                          legacy_numbering=args.legacy_numbering, link=args.link,
                          problems=args.problems)
    print(json.dumps({"status": "PASS", "files": results}, indent=2))


if __name__ == "__main__":
    main()
