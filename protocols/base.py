# protocols/base.py
"""Benchmark configuration for budgets, populations, seeds, and evaluation.

Protocols fix surrogate hyperparameters and sampling/penalty settings.
Methods and policies implement how those settings are used; configuring
a behavior here does not transfer its implementation to the protocol layer."""
from dataclasses import dataclass, field


@dataclass
class ProtocolConfig:
    name: str

    # Optimization budget and population size.
    pop_size: int
    maxgen: int

    # HF budget = init_points + (maxgen//update_interval)*hf_points_per_update.
    init_points: int
    update_interval: int
    hf_points_per_update: int

    # Shared constraint-handling parameters.
    penalty_min: float
    penalty_max: float

    # Default penalty mode uses the dynamic paper penalty schedule.
    # epsilon, feasibility, and decode support constraint-handling ablations.
    # Use a separate protocol file for ablations to preserve baseline settings.
    constraint_mode: str = "penalty"

    # Evaluation grid.
    mse_grid_nx: int = 200
    mse_grid_nt: int = 200
    mse_grid_nmu: int = None   # Parameter-axis grid size for F09+; None for nonparametric problems.

    # Explicit HF budget overrides the value computed from sampling settings.
    hf_query_budget: int = None

    # state counts labeled points (F01-F11 paper protocol).
    # parameter counts distinct parameter vectors.
    budget_kind: str = "state"

    # Sampling values belong to the protocol; behavior belongs to Method/Policy.
    update_policy: dict = field(default_factory=lambda: {"type": "periodic_topk"})

    # Per-surrogate hyperparameters, e.g. {"gp": {...}, "pigp": {...}, "pinn": {...}}.
    surrogates: dict = field(default_factory=dict)

    # Numerical overrides; None keeps problem/dataset defaults.
    # surrogate_query_dtype: float32 or float64; float64 avoids input rounding.
    # label_dtype: float32 or float64; float64 computes pool f/vio from unrounded labels.
    numerics: dict = field(default_factory=dict)

    # Method-specific settings for self-managed algorithms such as Ji SAEA and GLoSADE.
    # An empty mapping leaves defaults unchanged. load_protocol merges top-level
    # method_defaults with per-problem overrides.
    method_params: dict = field(default_factory=dict)

    # Optimizer registry key (de/cmaes/pso) for optimizer ablations.
    # Existing *_de methods select DE explicitly and do not read this field.
    # Composite methods can use this field to select a search operator.
    optimizer: str = "de"

    @property
    def hf_budget(self) -> int:
        """Return the protocol's total true-evaluation budget."""
        if self.hf_query_budget is not None:
            return int(self.hf_query_budget)
        return self.init_points + (self.maxgen // self.update_interval) * self.hf_points_per_update

    @classmethod
    def from_yaml_dict(cls, name: str, d: dict) -> "ProtocolConfig":
        d = dict(d)
        surrogates = d.pop("surrogates", None)
        update_policy = d.pop("update_policy", None)
        numerics = d.pop("numerics", None)
        method_params = d.pop("method_params", None)
        return cls(
            name=name,
            surrogates=surrogates or {},
            update_policy=update_policy or {"type": "periodic_topk"},
            numerics=numerics or {},
            method_params=method_params or {},
            **d,
        )
