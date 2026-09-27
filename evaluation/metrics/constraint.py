# evaluation/metrics/constraint.py
"""Feasibility and raw constraint-violation metrics.

Feasibility uses vio <= FEASIBLE_TOL (1e-4), rather than exact zero.
Keep raw violations so that feasibility can be recomputed at other tolerances."""
import numpy as np

FEASIBLE_TOL = 1e-4


def feasible_flags(violations, tol: float = FEASIBLE_TOL) -> np.ndarray:
    """Map violations to feasibility flags; NaN is infeasible."""
    v = np.asarray(violations, dtype=float)
    return np.isfinite(v) & (v <= tol)


def feasible_rate(violations, tol: float = FEASIBLE_TOL) -> float:
    """Fraction feasible among individuals or one representative violation per run."""
    v = np.asarray(violations, dtype=float)
    if v.size == 0:
        return float("nan")
    return float(feasible_flags(v, tol).mean())


def mean_violation(violations) -> float:
    """Mean raw violation, without tolerance clipping."""
    v = np.asarray(violations, dtype=float)
    return float(np.nanmean(v)) if v.size else float("nan")


def max_violation(violations) -> float:
    """Maximum raw violation."""
    v = np.asarray(violations, dtype=float)
    return float(np.nanmax(v)) if v.size else float("nan")
