# methods/pigp_de.py
"""PIGP-DE: PIGP field surrogate, DE search, periodic top-K updates and dynamic penalty."""
from surrogates.pigp import PIGPSurrogate
from methods.base import BenchmarkMethod, make_update_policy
from methods.policies import DynamicPenalty
from optimizers.de import DE


class PIGPDEMethod(BenchmarkMethod):
    name = "PIGP-DE"
    constraint_mode = "penalty"

    def __init__(self, protocol, seed: int):
        super().__init__(
            surrogate=PIGPSurrogate(),
            surrogate_config=protocol.surrogates.get("pigp", {}),
            update_policy=make_update_policy(protocol, seed),
            penalty=DynamicPenalty(protocol.penalty_min, protocol.penalty_max),
            seed=seed,
        )

    def build_optimizer(self, problem, protocol, fitness_fn, constraint_fn,
                        online_hook):
        return DE(
            bounds=problem.decision_bounds,
            pop_size=protocol.pop_size,
            max_gen=protocol.maxgen,
            fitness_function=fitness_fn,
            constraint=constraint_fn,
            penalty_min=self.penalty.min,
            penalty_max=self.penalty.max,
            ever_gen=self.update_policy.interval,
            online_hook=online_hook,
            constraint_mode=self.constraint_mode,
            seed=self.seed,
        )
