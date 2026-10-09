"""Check reference candidates by substitution and optional stored-field evaluation.

By default, substitute the recorded target states into the implemented
objective and constraints. This is not a PDE reachability/global-optimality
certificate. Supply --data-dir to also check available recorded decisions
against stored PDE fields. Large compressed fields are reported as unsupported
instead of silently allocating/decompressing several GB.
"""
import importlib
import json
from pathlib import Path
import struct
import sys
import zipfile

import numpy as np
from scipy.interpolate import RegularGridInterpolator

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
ORDER = ("f01", "f02", "f03", "f04", "f05", "f06", "f07", "f08", "f09", "f10", "f11")
SMALL_FIELD_BYTES = 64 * 1024**2


class ConstantState:
    """Target substitution, explicitly not a PDE solver or state surrogate."""

    def __init__(self, value):
        self.value = float(value)

    def evaluate(self, queries):
        return np.full(len(queries), self.value, dtype=float)


def read_header(stream):
    version = np.lib.format.read_magic(stream)
    if version == (1, 0):
        return np.lib.format.read_array_header_1_0(stream)
    if version == (2, 0):
        return np.lib.format.read_array_header_2_0(stream)
    raise ValueError(f"Unsupported NPY header version: {version}")


def stored_array(path, info):
    """Map an uncompressed NPY ZIP member without materializing its field."""
    if info.compress_type != zipfile.ZIP_STORED:
        raise ValueError("Sparse mapping requires an uncompressed ZIP member")
    with path.open("rb") as stream:
        stream.seek(info.header_offset)
        header = struct.unpack("<4s5H3I2H", stream.read(30))
        if header[0] != b"PK\x03\x04":
            raise ValueError("Invalid ZIP local header")
        stream.seek(header[-2] + header[-1], 1)
        shape, fortran, dtype = read_header(stream)
        offset = stream.tell()
    return np.memmap(path, mode="r", offset=offset, shape=shape, dtype=dtype,
                     order="F" if fortran else "C")


def archived_decision(code, entry):
    target = entry["target"]
    reach = entry.get("reachability_example", entry.get("reachability_examples", {}))
    coordinate = reach.get("t", reach.get("x2"))
    if isinstance(coordinate, list):
        coordinate = coordinate[0]
    if code == "f06":
        return [float(target["x"]), float(target["t"])]
    if code in ("f09", "f10", "f11") or coordinate is None:
        return None
    if code == "f07":
        x = np.pi
    elif code == "f08":
        x = np.sqrt(0.5)
    else:
        x = float(target.get("x", target.get("z1")))
    decision = [float(x), float(coordinate)]
    if code == "f02":
        decision.append(float(target["z3"]))
    return decision


def target_state(code, target, module):
    if code == "f08":
        return 0.5
    if code == "f06":
        x, t = float(target["x"]), float(target["t"])
        center = module.X_S0 + module.C_CONV * t
        return float(module._background_np(x, t)
                     + module.A_PKT * np.exp(-((x - center) / module.W_PKT)**2))
    return float(target.get("u", target.get("z2")))


def algebra_check(code, entry, problem, module, decision):
    target = entry["target"]
    expected = float(entry["f_star_decimal"])
    if code in ("f09", "f10", "f11"):
        u, j = float(target["u"]), float(target["J"])
        value = problem._F(u, j)
        result = {"u": u, "J": j, "objective": value, "constraints": []}
    else:
        u = target_state(code, target, module)
        provider = ConstantState(u)
        d = np.asarray(decision)
        if code == "f06":
            # Algebra-only mode has no dataset to attach. Evaluate the same
            # explicit formula without marking the dataset as validated.
            x, t = d
            value = float((1 - (u - module._background_np(x, t)) / module.A_PKT)**2
                          + module.ALPHA_T * (t - module.T_STAR)**2)
        else:
            value = problem.evaluate_fitness(d, provider)
        result = {"u": u, "objective": value,
                  "violation": problem.violation(d, provider),
                  "constraints": problem.constraint_components(d, provider)}
    result.update(objective_abs_error=abs(value - expected),
                  status="target_substituted_not_a_global_certificate",
                  pde_reachability_verified=False)
    return result


def field_point(path, problem, decision):
    """Use official small-field query or mathematically identical local RGI."""
    with zipfile.ZipFile(path) as archive:
        if "u.npy" not in archive.namelist():
            return {"status": "missing_array", "reason": "Dataset has no u.npy"}
        info = archive.getinfo("u.npy")
        with archive.open(info) as stream:
            shape, _, dtype = read_header(stream)
        description = {"stored_shape": list(shape), "stored_dtype": str(dtype),
                       "file_bytes": path.stat().st_size}
    if info.compress_type != zipfile.ZIP_STORED and info.file_size > SMALL_FIELD_BYTES:
        return dict(description, status="unsupported_large_compressed_field",
                    reason="Sparse checking cannot map compressed u.npy; full decompression was not requested.")
    problem.attach_data(str(path))
    query = np.asarray(problem.decision_to_query(np.asarray(decision)), dtype=float)
    if info.compress_type != zipfile.ZIP_STORED or problem.name == "F07":
        # F07 requires the official transpose/NaN-median/hole conventions.
        reference = problem.load_reference(str(path))
        value = float(reference.query(query.reshape(1, -1))[0])
        method = "official ReferenceDataset.query"
    else:
        with np.load(path) as data:
            if "x" not in data.files or "t" not in data.files:
                return dict(description, status="missing_array", reason="2D x/t axes are required")
            axes = [np.asarray(data[key], dtype=float) for key in ("x", "t")]
        if len(shape) != 2 or shape != tuple(len(axis) for axis in axes):
            return dict(description, status="unsupported_layout", reason="Expected u[x,t] 2D field")
        if any(len(axis) < 2 or not np.all(np.diff(axis) > 0) for axis in axes):
            return dict(description, status="unsupported_axes", reason="Sparse path requires ascending axes")
        query = np.clip(query, [axis[0] for axis in axes], [axis[-1] for axis in axes])
        indices = [max(0, min(len(axis) - 2, int(np.searchsorted(axis, query[k]) - 1)))
                   for k, axis in enumerate(axes)]
        field = stored_array(path, info)
        block = np.array(field[indices[0]:indices[0]+2, indices[1]:indices[1]+2], dtype=float)
        del field
        if not np.isfinite(block).all():
            return dict(description, status="unsupported_nonfinite_local_field",
                        reason="Sparse path does not invent a NaN fill convention")
        local_axes = tuple(axis[index:index+2] for axis, index in zip(axes, indices))
        value = float(RegularGridInterpolator(local_axes, block)(query.reshape(1, -1))[0])
        method = "read-only uncompressed NPZ mapping; local 2x2 linear RegularGridInterpolator"
    return dict(description, status="evaluated", u=value, method=method)


def verify(code, data_dir=None):
    """Return one candidate check; completion does not certify global optimality."""
    if code not in ORDER:
        raise ValueError(f"Unknown problem identifier: {code}")
    source = ROOT / "reference/targets" / f"{code}.json"
    if not source.is_file():
        raise FileNotFoundError(f"Missing reference record: {source}")
    if data_dir is not None:
        data_dir = Path(data_dir)
    incomplete = False
    entry = json.loads(source.read_text(encoding="utf-8"))
    module = importlib.import_module("problems." + code)
    problem = getattr(module, code.upper())()
    decision = archived_decision(code, entry)
    row = {"paper_problem": f"F{int(code[1:])}", "code_problem": code,
           "reference_record": entry, "archived_decision": decision,
           "decision_bounds": problem.decision_bounds.tolist(),
           "algebra": algebra_check(code, entry, problem, module, decision),
           "pde": {"status": "not_requested"}}
    if data_dir:
        path = data_dir / f"{code}.npz"
        if decision is None:
            row["pde"] = {"status": "missing_complete_pde_decision",
                          "reason": "Archive only supplies target U/J; reachability is not established."}
        elif not path.is_file():
            row["pde"] = {"status": "missing_dataset", "file": path.name}
        else:
            check = field_point(path, problem, decision)
            check["file"] = path.name
            if check["status"] == "evaluated":
                provider = ConstantState(check["u"])
                d = np.asarray(decision)
                check.update(target_state_abs_error=abs(check["u"] - row["algebra"]["u"]),
                             objective=problem.evaluate_fitness(d, provider),
                             violation=problem.violation(d, provider),
                             constraints=problem.constraint_components(d, provider))
                check["objective_abs_error_from_reference"] = abs(check["objective"] - float(entry["f_star_decimal"]))
                check["feasible_at_vio_1e_minus4"] = bool(check["violation"] <= 1e-4)
                check["global_optimality_certified"] = False
            row["pde"] = check
        incomplete |= row["pde"]["status"] != "evaluated"
    row.update(
        schema="reference-candidate-verification-v1",
        reference_source=str(source.relative_to(ROOT)),
        scope="Reference-candidate substitution and optional stored-field evaluation; no optimization or PDE solve.",
        precision_note="Float64 substitution errors do not certify stored decimal digits or global optimality.",
        mode="algebra_and_requested_pde_points" if data_dir else "algebra_only",
        requested_checks_complete=not incomplete,
        status="complete" if not incomplete else "incomplete")
    return row
