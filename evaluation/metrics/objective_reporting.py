"""Per-run reference-floor reporting with explicit tolerance attribution.

Raw objectives, decisions, violations and feasibility remain unchanged.
"""
import csv
from functools import lru_cache
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
REPORTING_NOTE = (
    "Objective tables use per-run F_reported=max(F_raw,F_ref) only when a "
    "sub-reference value has verified feasibility-tolerance attribution. "
    "Unverified cases retain F_raw and are listed separately. Means, sample "
    "variances and sample SDs use valid feasible F_reported values (ddof=1); "
    "fewer than two values give undefined variance/SD. Raw results and FR are unchanged.")


def finite_number(value):
    try:
        value = float(value)
        return value if np.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def is_feasible(value):
    return value is True or str(value).lower() == "true"


@lru_cache(maxsize=None)
def reference_value(problem):
    if problem == "f05":
        return -5.508013271595
    path = ROOT / "reference/targets" / f"{problem}.json"
    return float(json.loads(path.read_text())["f_star_decimal"]) if path.is_file() else None


@lru_cache(maxsize=1)
def confirmed_f5_runs():
    path = ROOT / "reproduction/metadata/f5_algebra_reporting.csv"
    with path.open(encoding="utf-8", newline="") as stream:
        return {r["history_sha256"]: r for r in csv.DictReader(stream)}


def annotate_run(row):
    """Append report fields; never infer tolerance attribution from CV alone."""
    raw = finite_number(row["best_f"])
    cv = finite_number(row.get("best_cv"))
    f_ref = reference_value(row["problem"])
    valid = is_feasible(row["feasible"]) and raw is not None
    basis = ""
    if valid and f_ref is not None and raw < f_ref:
        if row["problem"] == "f05":
            approved = confirmed_f5_runs().get(row.get("history_sha256", ""))
            if (approved and int(row["seed"]) == int(approved["seed"])
                    and raw == float(approved["F_raw"])
                    and cv == float(approved["CV_raw"]) and 0 < cv <= 1e-4):
                basis = approved["clipping_basis"]
        elif row["problem"] == "f08":
            x, state = map(finite_number, (row.get("best_x"), row.get("best_state")))
            if x is not None and state is not None and cv is not None:
                gap = state-x*x
                reconstructed = x*x+(state-1)**2
                if (reconstructed < .75 and 0 < gap <= .0011+1e-10 and 0 <= cv <= 1e-4
                        and np.isclose(raw, reconstructed, rtol=1e-10, atol=1e-10)):
                    basis = "F8 equality-gap identity checked against saved state/objective and existing tolerance band"
    below = valid and f_ref is not None and raw < f_ref
    clipped = bool(below and basis)
    status = ("no_valid_feasible_objective" if not valid else
              "verified_tolerance_clipped" if clipped else
              "below_reference_unverified" if below else "unchanged")
    return dict(row, F_raw=raw, F_reported=f_ref if clipped else raw if valid else None,
                F_ref=f_ref, CV_raw=cv, was_clipped=clipped,
                clipping_basis=basis, reporting_status=status)


def sample_moments(values):
    values = np.asarray([v for v in map(finite_number, values) if v is not None], float)
    n = len(values)
    constant = bool(n and np.all(values == values[0]))
    mean = float(values[0] if constant else values.mean()) if n else float("nan")
    variance = (0.0 if constant else float(values.var(ddof=1))) if n > 1 else float("nan")
    return dict(n=n, mean=mean, variance_ddof1=variance, std_ddof1=float(np.sqrt(variance)))


def objective_summary(rows):
    rows = [annotate_run(r) for r in rows]
    reported = sample_moments(r["F_reported"] for r in rows if is_feasible(r["feasible"]))
    raw = sample_moments(r["F_raw"] for r in rows if is_feasible(r["feasible"]))
    result = {"F_reported_"+k: v for k, v in reported.items()}
    # Keep established summary columns while making their reporting definition explicit.
    result.update(best_f_n=reported["n"], best_f_mean=reported["mean"],
                  best_f_std_ddof1=reported["std_ddof1"], best_f_variance_ddof1=reported["variance_ddof1"],
                  F_raw_mean=raw["mean"], F_raw_std_ddof1=raw["std_ddof1"],
                  clipped_runs=sum(r["was_clipped"] for r in rows),
                  unverified_below_reference_runs=sum(r["reporting_status"] == "below_reference_unverified" for r in rows))
    return result
