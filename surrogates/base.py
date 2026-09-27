# surrogates/base.py
"""Base interface for surrogates of the PDE state u(Q).

Q has shape (n, dim_query): (x,t) for F01-F08 and (x,t,parameter) for
F09-F11. Geometry and normalization bounds come from problem.query_bounds.
Training epochs, kernels and other hyperparameters belong to surrogate
protocol configuration rather than the problem definition.
"""
from abc import ABC, abstractmethod

import numpy as np


class Surrogate(ABC):
    """Common surrogate interface used by Experiment independently of model type."""

    name: str = "surrogate"

    @abstractmethod
    def setup(self, problem, seed: int, config: dict = None):
        """Create the untrained model using kernel/network/training protocol settings."""

    @abstractmethod
    def fit(self, Q: np.ndarray, u: np.ndarray, **ctx):
        """Fit the accumulated pool: Q has shape (n, dim_query), u has shape (n,).

        ctx carries training context such as gen/maxgen for PINN learning-rate
        scheduling. Context-independent models such as GP/PIGP ignore it.
        """

    @abstractmethod
    def predict(self, Q: np.ndarray) -> np.ndarray:
        """Predict Q (n, dim_query) -> u (n,); n=1 represents a single query."""
