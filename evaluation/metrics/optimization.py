# evaluation/metrics/optimization.py
"""Objective and success metrics from the consumed HF archive.

Do not score surrogate predictions or unconsumed monitoring points.
A run succeeds when CV <= 1e-4 and abs(F-F*) / max(1, abs(F*)) <= 1e-4.
F* denotes the feasible numerical reference candidate. Its verification scripts are in reference/."""
import numpy as np

from evaluation.metrics.constraint import FEASIBLE_TOL

SUCCESS_EPS = 1e-4


def best_feasible_objective(objs, vios, tol: float = FEASIBLE_TOL) -> float:
    """Minimum feasible objective in one HF archive; NaN if none is feasible."""
    o = np.asarray(objs, dtype=float)
    v = np.asarray(vios, dtype=float)
    feas = np.isfinite(v) & (v <= tol) & np.isfinite(o)
    if not feas.any():
        return float("nan")
    return float(o[feas].min())


def success_flags(objs, vios, f_star: float,
                  eps: float = SUCCESS_EPS, tol: float = FEASIBLE_TOL) -> np.ndarray:
    """Return per-run success flags from best feasible objectives and violations.

    objs and vios have shape (n_runs,). Use NaN for runs without a feasible
    solution; those runs are counted as unsuccessful."""
    o = np.asarray(objs, dtype=float)
    v = np.asarray(vios, dtype=float)
    rel = np.abs(o - f_star) / max(1.0, abs(f_star))
    return np.isfinite(o) & np.isfinite(v) & (v <= tol) & (rel <= eps)


def success_rate(objs, vios, f_star: float,
                 eps: float = SUCCESS_EPS, tol: float = FEASIBLE_TOL) -> float:
    """Fraction of successful runs."""
    o = np.asarray(objs, dtype=float)
    if o.size == 0:
        return float("nan")
    return float(success_flags(objs, vios, f_star, eps, tol).mean())
