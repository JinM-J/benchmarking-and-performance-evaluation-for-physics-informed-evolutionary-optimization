# optimizers/base.py
"""Common search interface with a run loop and online-update callback.

Preserve each optimizer's RNG ordering when modifying the interface."""
from abc import ABC, abstractmethod


class Optimizer(ABC):
    """Optimizer interface independent of concrete problem and surrogate types.

    Experiment constructs and injects the fitness and constraint callbacks."""

    name: str = "optimizer"

    @abstractmethod
    def run(self):
        """Run the search and return (population, fitness)."""
