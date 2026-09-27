"""PINO-DE: physics-informed FNO field surrogate with differential evolution."""
from surrogates.pino import PINOSurrogate
from methods.base import BenchmarkMethod, make_update_policy
from methods.policies import DynamicPenalty
from optimizers.de import DE


class PINODEMethod(BenchmarkMethod):
    name = "PINO-DE"
    constraint_mode = "penalty"

    def __init__(self, protocol, seed: int):
        config = protocol.surrogates.get("pino", {})
        super().__init__(
            surrogate=PINOSurrogate(config),
            surrogate_config=config,
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
