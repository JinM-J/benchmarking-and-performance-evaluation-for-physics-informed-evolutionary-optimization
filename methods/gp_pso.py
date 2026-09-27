# methods/gp_pso.py
"""GP-PSO: GP field surrogate, PSO search, periodic top-K updates and dynamic penalty.

Optimizer ablation: surrogate, sampling and penalty match the DE variant.
Uses optimizers/pso.py defaults: w 0.9 to 0.4, c1=c2=2.0, v_clamp=0.2."""
from surrogates.gp import GPSurrogate
from methods.base import BenchmarkMethod, make_update_policy
from methods.policies import DynamicPenalty
from optimizers.pso import PSOOptimizer


class GPPSOMethod(BenchmarkMethod):
    name = "GP-PSO"

    def __init__(self, protocol, seed: int):
        super().__init__(
            surrogate=GPSurrogate(),
            surrogate_config=protocol.surrogates.get("gp", {}),
            update_policy=make_update_policy(protocol, seed),
            penalty=DynamicPenalty(protocol.penalty_min, protocol.penalty_max),
            seed=seed,
        )

    def build_optimizer(self, problem, protocol, fitness_fn, constraint_fn,
                        online_hook):
        return PSOOptimizer(
            bounds=problem.decision_bounds,
            pop_size=protocol.pop_size,
            max_gen=protocol.maxgen,
            fitness_function=fitness_fn,
            constraint=constraint_fn,
            penalty_min=self.penalty.min,
            penalty_max=self.penalty.max,
            ever_gen=self.update_policy.interval,
            online_hook=online_hook,
            seed=self.seed,
        )
