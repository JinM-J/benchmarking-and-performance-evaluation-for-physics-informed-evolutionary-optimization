# optimizers/pso.py
"""Global-best particle swarm optimization for optimizer ablations.

Share the DE constructor interface and vectorized fitness/constraint
callbacks: X (N, dim) -> (N,). Single queries use atleast_2d.
Update v_i = w*v_i + c1*r1*(pbest_i-x_i) + c2*r2*(gbest-x_i),
then x_i = x_i + v_i. Inertia decreases from w_max to w_min;
velocity is limited relative to the box width.
Use the same dynamic f(x) + penalty(gen)*violation(x) as DE.
Clip positions to bounds and zero velocity components that hit a bound
to prevent persistent outward motion at the boundary."""
import numpy as np
from dataclasses import dataclass, field
from typing import Callable, Optional

from optimizers.base import Optimizer


@dataclass
class PSOOptimizer(Optimizer):
    bounds: np.ndarray = None
    pop_size: int = 100
    max_gen: int = 1000

    seed: int = 42

    penalty_min: float = 10.0
    penalty_max: float = 100.0

    # Vectorized population callbacks: X (N, dim) -> (N,), matching DE.
    constraint: Optional[Callable[[np.ndarray], float]] = None
    fitness_function: Optional[Callable[[np.ndarray], float]] = None

    # Surrogate-update hook at the same ever_gen cadence as DE.
    ever_gen: int = 5
    online_hook: Optional[Callable[["PSOOptimizer", int], None]] = None

    # PSO parameters.
    w_max: float = 0.9       # Initial inertia for exploration.
    w_min: float = 0.4       # Final inertia for refinement.
    c1: float = 2.0          # Cognitive coefficient.
    c2: float = 2.0          # Social coefficient.
    v_clamp: float = 0.2     # Velocity limit as a fraction of box width.

    name: str = field(default="PSO", init=False)
    dim: int = field(init=False)
    rng: np.random.Generator = field(init=False)

    def __post_init__(self):
        self.bounds = np.asarray(self.bounds, dtype=float)
        self.dim = self.bounds.shape[0]
        self.rng = np.random.default_rng(self.seed)
        span = self.bounds[:, 1] - self.bounds[:, 0]
        self.v_max = self.v_clamp * span

        self.pop = self.rng.uniform(self.bounds[:, 0], self.bounds[:, 1],
                                    (self.pop_size, self.dim))
        self.vel = self.rng.uniform(-self.v_max, self.v_max,
                                    (self.pop_size, self.dim))
        self.fitness = None
        # Personal and global bests are initialized after gen=0 evaluation.
        self.pbest_pos = self.pop.copy()
        self.pbest_fit = None
        self.gbest_pos = None

    def _penalty(self, gen: int) -> float:
        # Use DE's linear penalty schedule for comparable optimizer ablations.
        return self.penalty_min + (self.penalty_max - self.penalty_min) * (gen / self.max_gen)

    def evaluate(self, gen: int):
        """Evaluate current positions using f + penalty*violation."""
        penalty = self._penalty(gen)
        # One vectorized population call through the experiment batch path.
        f = np.asarray(self.fitness_function(self.pop), dtype=float)
        c = np.asarray(self.constraint(self.pop), dtype=float)
        self.fitness = f + penalty * c

    def _reeval_pbest(self, gen: int):
        """Reevaluate historical best positions after the surrogate is refitted.

        Old pbest/gbest fitness values are not comparable to predictions from
        the updated surrogate; refresh them before selecting new bests."""
        penalty = self._penalty(gen)
        f = np.asarray(self.fitness_function(self.pbest_pos), dtype=float)
        c = np.asarray(self.constraint(self.pbest_pos), dtype=float)
        self.pbest_fit = f + penalty * c

    def run(self):
        # The initial hook samples data and fits the surrogate, as in DE.
        if self.online_hook is not None:
            self.online_hook(self, 0)

        self.evaluate(0)
        self.pbest_fit = self.fitness.copy()
        self.gbest_pos = self.pbest_pos[int(np.argmin(self.pbest_fit))].copy()

        for gen in range(1, self.max_gen + 1):
            w = self.w_max - (self.w_max - self.w_min) * (gen / self.max_gen)

            r1 = self.rng.random((self.pop_size, self.dim))
            r2 = self.rng.random((self.pop_size, self.dim))
            self.vel = (w * self.vel
                        + self.c1 * r1 * (self.pbest_pos - self.pop)
                        + self.c2 * r2 * (self.gbest_pos - self.pop))
            np.clip(self.vel, -self.v_max, self.v_max, out=self.vel)

            self.pop = self.pop + self.vel
            # Clip positions and zero clipped velocity components to avoid boundary sticking.
            below = self.pop < self.bounds[:, 0]
            above = self.pop > self.bounds[:, 1]
            self.pop = np.clip(self.pop, self.bounds[:, 0], self.bounds[:, 1])
            self.vel[below | above] = 0.0

            self.evaluate(gen)

            # Update personal and global bests using penalized fitness.
            better = self.fitness < self.pbest_fit
            self.pbest_pos[better] = self.pop[better]
            self.pbest_fit[better] = self.fitness[better]
            self.gbest_pos = self.pbest_pos[int(np.argmin(self.pbest_fit))].copy()

            # Trigger the online hook every ever_gen generations, as in DE.
            if self.online_hook is not None and (gen % self.ever_gen == 0):
                self.online_hook(self, gen)
                self.evaluate(gen)        # Reevaluate current positions after the surrogate changes.
                self._reeval_pbest(gen)   # Refresh historical-best fitness values under the new surrogate.
                self.gbest_pos = self.pbest_pos[
                    int(np.argmin(self.pbest_fit))].copy()

        # Inject gbest into the worst final-population slot for final_real diagnostics;
        # the historical best need not still be among current positions.
        worst = int(np.argmax(self.fitness))
        self.pop[worst] = self.gbest_pos
        self.fitness[worst] = (
            float(self.fitness_function(self.gbest_pos))
            + self._penalty(self.max_gen)
            * float(self.constraint(self.gbest_pos)))

        return self.pop, self.fitness
