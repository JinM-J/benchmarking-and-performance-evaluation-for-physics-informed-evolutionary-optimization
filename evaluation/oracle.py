# evaluation/oracle.py
"""Self-counting interface for true state queries during optimization.

n_state_queries counts one label per (x,t[,parameter]) row. The auxiliary
n_parameter_queries sums distinct parameter values within each query batch;
it is not a count of globally distinct solves. Ground truth for benchmark
fields is interpolated, so one state query is not one PDE solve.
Metric-only ReferenceDataset queries do not consume the algorithm's FE budget."""
import time

import numpy as np


class HighFidelityOracle:
    def __init__(self, reference, parameter_axes=None):
        self.reference = reference
        # Parameter axes: None selects axis 2 when the query dimension is at least 3.
        # Extended problems may declare explicit axes for complete parameter vectors.
        self.parameter_axes = (tuple(parameter_axes)
                               if parameter_axes is not None else None)
        self.n_state_queries = 0       # Number of queried HF state labels.
        self.n_parameter_queries = 0   # Sum of within-call distinct parameter counts.
        self.n_calls = 0               # Number of evaluate() calls.
        self.wall_time = 0.0           # Cumulative query time in seconds.

    # Paper FE alias: number of state queries.
    @property
    def n_hf_queries(self) -> int:
        return self.n_state_queries

    def evaluate(self, Q: np.ndarray) -> np.ndarray:
        """Evaluate Q of shape (n, dim_query); return true states of shape (n,)."""
        Q = np.asarray(Q, dtype=float)
        if Q.ndim == 1:
            Q = Q.reshape(1, -1)

        t0 = time.perf_counter()
        u = self.reference.query(Q).reshape(-1)
        self.wall_time += time.perf_counter() - t0

        self.n_state_queries += int(Q.shape[0])
        self.n_calls += 1
        if self.parameter_axes is not None:
            self.n_parameter_queries += int(
                np.unique(Q[:, list(self.parameter_axes)], axis=0).shape[0])
        elif Q.shape[1] >= 3:
            self.n_parameter_queries += int(np.unique(Q[:, 2]).size)
        return u
