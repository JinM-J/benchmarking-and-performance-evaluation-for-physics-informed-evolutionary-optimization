# optimizers/sade_de.py
"""Ji et al. DE search operator for SaDE-SA-GRM.

Use feasibility-first comparison: feasible dominates infeasible; compare
objectives when both are feasible, otherwise compare CV, where
CV = sum(max(g,0)) + sum(max(abs(h)-epsilon,0)).
Each member randomly selects best/2, rand/2, rand-to-best/1,
current-to-best/1, or current-to-rand/1, with binomial crossover.
Draw F = 0.5*rand + 0.5 and CR = 0.5*rand + 0.5 per member per generation.
Injected evaluation uses RBFN predictions in the Ji loop, without HF calls.
One persistent default_rng(seed) stream controls the search."""
import numpy as np


class SADE_DE:
    """Ji DE surrogate search with injected objective and CV predictions."""

    def __init__(self, bounds, pop_size: int, evaluate_fn, seed: int,
                 eq_tol: float = 1e-4, initial_pop=None):
        self.bounds = np.asarray(bounds, dtype=float)
        self.pop_size = int(pop_size)
        self.dim = self.bounds.shape[0]
        self.evaluate_fn = evaluate_fn   # X (n,d) -> (f_hat (n,), cv_hat (n,))
        self.rng = np.random.default_rng(seed)
        self.eq_tol = float(eq_tol)
        if self.pop_size < 6:
            raise ValueError("Ji SaDE rand/2 requires pop_size >= 6")
        if initial_pop is None:
            self.pop = self.rng.uniform(self.bounds[:, 0], self.bounds[:, 1],
                                        (self.pop_size, self.dim))
        else:
            initial_pop = np.asarray(initial_pop, dtype=float)
            if initial_pop.shape != (self.pop_size, self.dim):
                raise ValueError(
                    "initial_pop must have shape "
                    f"({self.pop_size}, {self.dim}); received {initial_pop.shape}"
                )
            self.pop = np.clip(initial_pop.copy(), self.bounds[:, 0],
                               self.bounds[:, 1])
        self.f, self.cv = self._eval(self.pop)

    # ------------------------------------------------------------------
    def _eval(self, X):
        return self.evaluate_fn(X)

    def _better(self, f1, cv1, f2, cv2) -> bool:
        """Return whether candidate 1 dominates candidate 2 under feasibility-first selection."""
        f1_feas = cv1 <= 0.0
        f2_feas = cv2 <= 0.0
        if f1_feas != f2_feas:
            return f1_feas
        if f1_feas and f2_feas:
            return f1 < f2
        return cv1 < cv2

    def _best_index(self) -> int:
        """Return the best member under the feasibility rule, even if none is feasible."""
        best = 0
        for j in range(1, self.pop_size):
            if self._better(self.f[j], self.cv[j],
                            self.f[best], self.cv[best]):
                best = j
        return best

    def _mutate(self, i, scale):
        """Randomly select one of five strategies and return its mutant vector."""
        n = self.pop_size
        idx = [j for j in range(n) if j != i]
        strategy = self.rng.integers(0, 5)
        if strategy == 0:      # DE/best/2/bin
            best = self._best_index()
            r = self.rng.choice(idx, 4, replace=False)
            return (self.pop[best] + scale * (self.pop[r[0]] - self.pop[r[1]])
                    + scale * (self.pop[r[2]] - self.pop[r[3]]))
        if strategy == 1:      # DE/rand/2/bin
            r = self.rng.choice(idx, 5, replace=False)
            return (self.pop[r[0]] + scale * (self.pop[r[1]] - self.pop[r[2]])
                    + scale * (self.pop[r[3]] - self.pop[r[4]]))
        if strategy == 2:      # DE/rand-to-best/1/bin
            best = self._best_index()
            r = self.rng.choice(idx, 3, replace=False)
            return (self.pop[r[0]] + scale * (self.pop[best] - self.pop[r[0]])
                    + scale * (self.pop[r[1]] - self.pop[r[2]]))
        if strategy == 3:      # DE/current-to-best/1/bin
            best = self._best_index()
            r = self.rng.choice(idx, 2, replace=False)
            return (self.pop[i] + scale * (self.pop[best] - self.pop[i])
                    + scale * (self.pop[r[0]] - self.pop[r[1]]))
        # strategy == 4:       DE/current-to-rand/1
        r = self.rng.choice(idx, 3, replace=False)
        return (self.pop[i] + scale * (self.pop[r[0]] - self.pop[i])
                + scale * (self.pop[r[1]] - self.pop[r[2]]))

    def _F(self):
        return 0.5 * self.rng.random() + 0.5

    def _CR(self):
        return 0.5 * self.rng.random() + 0.5

    def evolve(self, n_gen: int):
        """Evolve on surrogate predictions for n_gen generations without HF calls."""
        for _ in range(int(n_gen)):
            for i in range(self.pop_size):
                scale = self._F()
                mutant = np.clip(self._mutate(i, scale), self.bounds[:, 0],
                                 self.bounds[:, 1])
                cross = self.rng.random(self.dim) < self._CR()
                if not np.any(cross):
                    cross[self.rng.integers(0, self.dim)] = True
                trial = np.where(cross, mutant, self.pop[i])
                f_t, cv_t = self._eval(trial.reshape(1, -1))
                if self._better(f_t[0], cv_t[0], self.f[i], self.cv[i]):
                    self.pop[i] = trial
                    self.f[i] = f_t[0]
                    self.cv[i] = cv_t[0]

    def candidates(self) -> np.ndarray:
        """Return the current population for consensus selection in the Ji loop."""
        return self.pop.copy()
