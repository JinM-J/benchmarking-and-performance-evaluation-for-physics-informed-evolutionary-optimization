# evaluation/metrics/__init__.py
"""Optimization, constraint and surrogate-quality metrics.

Grid MSE monitors training; surrogate metrics include PDE residuals where
derivatives are available. Optimization scores use consumed HF archives.
FEASIBLE_TOL = 1e-4; success additionally requires
abs(F-F*) / max(1, abs(F*)) <= 1e-4. F* is a numerical reference candidate.
Raw violations are retained separately from tolerance-based flags."""
from evaluation.metrics.grid import MetricGrid, mse_rmse_on_real_grid
from evaluation.metrics.constraint import (
    FEASIBLE_TOL, feasible_flags, feasible_rate, mean_violation, max_violation,
)
from evaluation.metrics.optimization import (
    SUCCESS_EPS, best_feasible_objective, success_flags, success_rate,
)

__all__ = [
    "MetricGrid", "mse_rmse_on_real_grid",
    "FEASIBLE_TOL", "feasible_flags", "feasible_rate",
    "mean_violation", "max_violation",
    "SUCCESS_EPS", "best_feasible_objective", "success_flags", "success_rate",
]
