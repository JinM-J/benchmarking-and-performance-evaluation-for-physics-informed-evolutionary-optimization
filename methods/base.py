# methods/base.py
"""Compose an optimization method from a surrogate, optimizer and update policy.

GP-DE, PINN-DE and PIGP-DE combine their respective field surrogates with
DE, PeriodicTopK and DynamicPenalty. Methods select HF queries, retraining
and constraint handling; protocols define budgets and comparison rules.
"""
from abc import ABC, abstractmethod

from surrogates.base import Surrogate
from methods.policies import (
    PeriodicTopK,
    TopKFullDecision,
    TopKMuRandomXT,
    TopKThetaBlock,
    DynamicPenalty,
)


def make_update_policy(protocol, seed: int):
    """Build the sampling policy selected by protocol.update_policy.type."""
    policy_cfg = getattr(protocol, "update_policy", None) or {}
    ptype = policy_cfg.get("type", "periodic_topk")
    if ptype == "periodic_topk":
        return PeriodicTopK(
            interval=protocol.update_interval,
            k=protocol.hf_points_per_update,
            init_points=protocol.init_points,
            seed=seed,
            # Candidate oversampling for domain filtering; default 1 preserves the baseline.
            candidate_factor=policy_cfg.get("candidate_factor", 1),
        )
    if ptype == "mu_topk_random_xt":
        return TopKMuRandomXT(
            interval=protocol.update_interval,
            k=protocol.hf_points_per_update,
            init_points=protocol.init_points,
            seed=seed,
            xt_per_mu=int(policy_cfg["xt_per_mu"]),
            # joint_random: paper sampling; hierarchical: stratified F09-F11 initialization.
            init_mode=policy_cfg.get("init_mode", "joint_random"),
        )
    if ptype == "topk_full_decision":
        return TopKFullDecision(
            interval=protocol.update_interval,
            k=protocol.hf_points_per_update,
            init_points=protocol.init_points,
            seed=seed,
        )
    if ptype == "theta_block":
        # Top-K parameter values with complete query blocks and interior points.
        return TopKThetaBlock(
            interval=protocol.update_interval,
            k=protocol.hf_points_per_update,
            init_points=protocol.init_points,
            seed=seed,
            interior_points=int(policy_cfg["interior_points"]),
        )
    raise KeyError(f"Unknown sampling policy type: {ptype}")


class BenchmarkMethod(ABC):
    """Container for a complete optimization method."""

    name: str = "method"

    def __init__(self, surrogate: Surrogate, surrogate_config: dict,
                 update_policy: PeriodicTopK, penalty: DynamicPenalty, seed: int):
        self.surrogate = surrogate
        self.surrogate_config = surrogate_config or {}
        self.update_policy = update_policy
        self.penalty = penalty
        self.seed = seed

    def setup(self, problem):
        """Initialize the surrogate without training, using protocol hyperparameters."""
        self.surrogate.setup(problem, self.seed, self.surrogate_config)

    @abstractmethod
    def build_optimizer(self, problem, protocol, fitness_fn, constraint_fn,
                        online_hook):
        """Build the optimizer using self.penalty; its internal schedule preserves DE RNG order."""
