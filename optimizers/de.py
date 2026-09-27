# optimizers/de.py
"""DE/best/rand/1 variant with reproducible random-number ordering.

F decreases linearly and CR increases linearly with generation.
penalty(gen) = penalty_min + (penalty_max-penalty_min)*(gen/max_gen).
Elitism replaces the worst member with the best; periodic restarts reset
a population fraction. online_hook runs every ever_gen generations.
Preserve the RNG consumption order for reference-run reproducibility."""
import numpy as np
from dataclasses import dataclass, field
from typing import Callable, Optional

from optimizers.base import Optimizer


# Takahama-Sakai epsilon-level control used in constraint ablations.
# epsilon_0 uses rank ceil(theta_frac*N) of descending initial violations.
# Epsilon decays with power cp until control_frac*T, then becomes exactly zero.
EPSILON_THETA_FRAC = 0.05
EPSILON_CONTROL_FRAC = 0.80
EPSILON_POWER = 5.0


@dataclass
class DE(Optimizer):
    bounds: np.ndarray = None
    pop_size: int = 100
    max_gen: int = 1000

    # Generation-dependent F and CR.
    F: float = 0.5
    F_max: float = 1.0
    CR: float = 0.4
    CR_max: float = 0.9

    seed: int = 42

    penalty_min: float = 10.0
    penalty_max: float = 100.0

    # Constraint modes affect ranking only and consume no RNG draws:
    # penalty: dynamic f + penalty*c (paper schedule increases from 10 to 100).
    # epsilon: compare f if c <= epsilon(gen), otherwise compare c (Takahama-Sakai).
    # feasibility: compare f if c <= feas_tol, otherwise compare c (Deb, 2000).
    # decode: DeCODE-inspired per-member weights eta_i with decreasing eta_max.
    constraint_mode: str = "penalty"
    feas_tol: float = 1e-4             # Feasibility threshold, equal to the evaluation FEASIBLE_TOL.
    eps_theta_frac: float = EPSILON_THETA_FRAC
    eps_control_frac: float = EPSILON_CONTROL_FRAC
    eps_power: float = EPSILON_POWER   # Epsilon decay exponent cp.
    big_key: float = 1e10              # Infeasible-key offset; must exceed feasible objective differences.
    eta_max0: float = 100.0            # Initial DeCODE weight upper bound, on the penalty_max scale.

    # Vectorized X (N, dim) -> (N,); also accepts a single (dim,) row.
    constraint: Optional[Callable[[np.ndarray], np.ndarray]] = None
    fitness_function: Optional[Callable[[np.ndarray], np.ndarray]] = None

    # online hook
    ever_gen: int = 5
    online_hook: Optional[Callable[["DE", int], None]] = None

    # elite/remake
    remake: int = 200
    re: float = 0.2

    name: str = field(default="DE", init=False)
    dim: int = field(init=False)
    rng: np.random.Generator = field(init=False)

    def __post_init__(self):
        self.bounds = np.asarray(self.bounds, dtype=float)
        self.dim = self.bounds.shape[0]
        self.rng = np.random.default_rng(self.seed)
        self.pop_init()
        self.fitness = None
        self._eps0 = None       # Lazy epsilon_0 from initial-population violation order statistics at gen=0.
        self._parent_f = None   # Cache raw parent objectives to avoid comparing keys from different epsilon levels.
        self._parent_c = None   # Cache raw parent surrogate violations for epsilon selection.

    def pop_init(self):
        self.pop = self.rng.uniform(self.bounds[:, 0], self.bounds[:, 1], (self.pop_size, self.dim))

    def _penalty(self, gen: int) -> float:
        return self.penalty_min + (self.penalty_max - self.penalty_min) * (gen / self.max_gen)

    def _eta_max(self, gen: int) -> float:
        """Decrease the DeCODE weight cap to zero, shifting exploration toward feasibility."""
        return self.eta_max0 * max(0.0, 1.0 - gen / self.max_gen)

    def _init_epsilon(self, c: np.ndarray) -> None:
        """Initialize epsilon_0 from initial violation order statistics."""
        if self._eps0 is not None:
            return
        violations = np.sort(np.asarray(c, dtype=float).reshape(-1))[::-1]
        if violations.size == 0:
            raise ValueError("Cannot initialize epsilon_0 from an empty population")
        theta = max(1, int(np.ceil(self.eps_theta_frac * violations.size)))
        theta = min(theta, violations.size)
        self._eps0 = float(max(violations[theta - 1], self.feas_tol))

    def _epsilon(self, gen: int) -> float:
        """Return the generation's epsilon, exactly zero after control generation Tc."""
        if self._eps0 is None:
            raise RuntimeError("epsilon_0 is uninitialized; evaluate the initial population first")
        control_gen = max(1, int(round(self.eps_control_frac * self.max_gen)))
        if gen >= control_gen:
            return 0.0
        return self._eps0 * (1.0 - gen / control_gen) ** self.eps_power

    def _epsilon_better(self, trial_f: np.ndarray, trial_c: np.ndarray,
                        parent_f: np.ndarray, parent_c: np.ndarray,
                        gen: int) -> np.ndarray:
        """Compare each trial with its target parent at the same generation's epsilon."""
        eps = self._epsilon(gen)
        both_relaxed = (trial_c <= eps) & (parent_c <= eps)
        equal_violation = trial_c == parent_c
        return np.where(both_relaxed | equal_violation,
                        trial_f < parent_f,
                        trial_c < parent_c)

    def _select_epsilon_trials(self, trials: np.ndarray,
                               trial_f: np.ndarray, trial_c: np.ndarray,
                               gen: int) -> None:
        """Recompute parent and trial keys at the current epsilon before targetwise selection."""
        # Parent f/c remain fixed between surrogate fits; update only generation-dependent keys.
        # Avoid redundant surrogate inference while keeping parent and trial keys comparable.
        self.fitness = self._fitness_key(self._parent_f, self._parent_c, gen)
        fit_trials = self._fitness_key(trial_f, trial_c, gen)
        better = self._epsilon_better(
            trial_f, trial_c, self._parent_f, self._parent_c, gen)
        for i in range(self.pop_size):
            if better[i]:
                self.pop[i] = trials[i]
                self._parent_f[i] = trial_f[i]
                self._parent_c[i] = trial_c[i]
                self.fitness[i] = fit_trials[i]

    def _fitness_key(self, f: np.ndarray, c: np.ndarray, gen: int) -> np.ndarray:
        """Return deterministic constraint-ranking keys of shape (N,), without RNG draws.

        Selection and elitism share these keys:
        - penalty: f + penalty(gen)*c, preserving reference arithmetic order.
        - feasibility: f when c <= feas_tol, otherwise big_key + c.
        - epsilon: f when c <= epsilon(gen), otherwise big_key + c.
          epsilon_0 is descending initial violation rank ceil(theta_frac*N),
          bounded below by feas_tol. For Tc = control_frac*max_gen,
          epsilon(gen) = epsilon_0*(1-gen/Tc)^cp before Tc and zero afterward.
        - decode: member i uses eta_i = eta_max(gen)*i/(N-1) and f + eta_i*c.
          Small weights emphasize feasible refinement; large weights relax
          constraints for exploration. Trials inherit their target's weight,
          giving a DeCODE-inspired subproblem-specific comparison."""
        if self.constraint_mode == "penalty":
            return f + self._penalty(gen) * c
        if self.constraint_mode == "feasibility":
            return np.where(c <= self.feas_tol, f, self.big_key + c)
        if self.constraint_mode == "epsilon":
            self._init_epsilon(c)
            eps = self._epsilon(gen)
            return np.where(c <= eps, f, self.big_key + c)
        if self.constraint_mode == "decode":
            n = f.shape[0]
            eta = self._eta_max(gen) * (np.arange(n) / max(n - 1, 1))
            return f + eta * c
        raise ValueError(f"Unknown constraint_mode={self.constraint_mode!r}; "
                         f"choose penalty/epsilon/feasibility/decode")

    def _F_CR(self, gen: int):
        F_now = self.F + (self.F_max - self.F) * (1 - gen / self.max_gen)
        CR_now = self.CR + (self.CR_max - self.CR) * (gen / self.max_gen)
        return float(F_now), float(CR_now)

    def evaluate(self, gen: int):
        f = np.asarray(self.fitness_function(self.pop), dtype=float)   # (N,)
        c = np.asarray(self.constraint(self.pop), dtype=float)         # (N,)
        if self.constraint_mode == "epsilon":
            self._parent_f = f.copy()
            self._parent_c = c.copy()
        self.fitness = self._fitness_key(f, c, gen)

    def run(self):
    # The initial hook samples labels and fits the surrogate before evaluation.
        if self.online_hook is not None:
            self.online_hook(self, 0)   # The hook must perform both sampling and surrogate fitting.

    # The surrogate is available for initial evaluation.
        self.evaluate(gen=0)

        for gen in range(1, self.max_gen + 1):
            F_now, CR_now = self._F_CR(gen)

            # Preserve the per-member RNG order during mutation and crossover.
            trials = np.empty_like(self.pop)
            for i in range(self.pop_size):
                idx = [j for j in range(self.pop_size) if j != i]
                r1, r2, r3 = self.rng.choice(idx, 3, replace=False)

                mutant = self.pop[r1] + F_now * (self.pop[r2] - self.pop[r3])
                mutant = np.clip(mutant, self.bounds[:, 0], self.bounds[:, 1])

                cross = self.rng.random(self.dim) < CR_now
                if not np.any(cross):
                    cross[self.rng.integers(0, self.dim)] = True

                trials[i] = np.where(cross, mutant, self.pop[i])

            # Evaluate all trials in a batch. sklearn batch and pointwise predictions
            # can differ by about one ULP; identical batch execution is deterministic.
            trial_f = np.asarray(self.fitness_function(trials), dtype=float)
            trial_c = np.asarray(self.constraint(trials), dtype=float)
            if self.constraint_mode == "epsilon":
                self._select_epsilon_trials(trials, trial_f, trial_c, gen)
            else:
                fit_trials = self._fitness_key(trial_f, trial_c, gen)

                # Non-epsilon selection retains reference-run arithmetic.
                for i in range(self.pop_size):
                    if fit_trials[i] < self.fitness[i]:
                        self.pop[i] = trials[i]
                        self.fitness[i] = fit_trials[i]

            # Elitism.
            best = int(np.argmin(self.fitness))
            worst = int(np.argmax(self.fitness))
            elite = self.pop[best].copy()
            elite_fit = float(self.fitness[best])

            self.pop[worst] = elite
            self.fitness[worst] = elite_fit
            if self.constraint_mode == "epsilon":
                self._parent_f[worst] = self._parent_f[best]
                self._parent_c[worst] = self._parent_c[best]

            # Periodic population restart.
            if self.remake and (gen % self.remake == 0):
                n_re = int(self.re * self.pop_size)
                cand = [k for k in range(self.pop_size) if k != worst]
                re_idx = self.rng.choice(cand, n_re, replace=False)
                self.pop[re_idx] = self.rng.uniform(self.bounds[:, 0], self.bounds[:, 1], (n_re, self.dim))
                self.evaluate(gen)  # Recompute fitness after resetting members.

            # Trigger surrogate updates every ever_gen generations.
            if self.online_hook is not None and (gen % self.ever_gen == 0):
                self.online_hook(self, gen)
                self.evaluate(gen)

        return self.pop, self.fitness
