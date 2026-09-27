# optimizers/cmaes.py
"""Covariance matrix adaptation evolution strategy for optimizer ablations.

Share DE's constructor and vectorized X (N, dim) -> (N,) fitness/constraint
callbacks, with atleast_2d support for single points.
Use Hansen's (mu/lambda, sigma) CMA-ES with rank-mu and rank-one covariance
updates, evolution paths p_sigma/p_c, and adaptive step size.
Constraints use DE's dynamic f(x) + penalty(gen)*violation(x).
Clip sampled points to box bounds and use clipped points for mean and
covariance updates, keeping the mean inside the box.
The initial step size is isotropic, so disparate coordinate scales can
degrade performance. Coordinate normalization belongs to the method layer."""
import numpy as np
from dataclasses import dataclass, field
from typing import Callable, Optional

from optimizers.base import Optimizer


@dataclass
class CMAESOptimizer(Optimizer):
    bounds: np.ndarray = None
    pop_size: int = 100          # lambda: number of samples per generation.
    max_gen: int = 1000

    seed: int = 42

    penalty_min: float = 10.0
    penalty_max: float = 100.0

    # Vectorized population callbacks: X (N, dim) -> (N,), matching DE.
    constraint: Optional[Callable[[np.ndarray], float]] = None
    fitness_function: Optional[Callable[[np.ndarray], float]] = None

    # Surrogate-update hook at the same ever_gen cadence as DE.
    ever_gen: int = 5
    online_hook: Optional[Callable[["CMAESOptimizer", int], None]] = None

    # Standard CMA-ES parameters; sigma0 is relative to the box width.
    sigma0: float = 0.3

    name: str = field(default="CMAES", init=False)
    dim: int = field(init=False)
    rng: np.random.Generator = field(init=False)

    def __post_init__(self):
        self.bounds = np.asarray(self.bounds, dtype=float)
        self.dim = self.bounds.shape[0]
        self.rng = np.random.default_rng(self.seed)

        lam, dim = self.pop_size, self.dim
        mu = lam // 2
        # Log weights for the top mu members; normalized weights define mu_eff.
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / w.sum()
        self.lam, self.mu, self.w = lam, mu, w
        self.mu_eff = 1.0 / float(w @ w)

        # Strategy parameters from Hansen's standard formulas.
        self.c_sigma = (self.mu_eff + 2) / (dim + self.mu_eff + 5)
        self.d_sigma = (1 + 2 * max(0.0, np.sqrt((self.mu_eff - 1) / (dim + 1)) - 1)
                        + self.c_sigma)
        self.c_c = (4 + self.mu_eff / dim) / (dim + 4 + 2 * self.mu_eff / dim)
        self.c_1 = 2 / ((dim + 1.3) ** 2 + self.mu_eff)
        self.c_mu = min(1 - self.c_1,
                        2 * (self.mu_eff - 2 + 1 / self.mu_eff)
                        / ((dim + 2) ** 2 + self.mu_eff))
        # Approximate E||N(0,I)|| for step-size normalization.
        self.chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        # Seeded uniform initial mean; initial step size scales with box width.
        span = self.bounds[:, 1] - self.bounds[:, 0]
        self.mean = self.rng.uniform(self.bounds[:, 0], self.bounds[:, 1])
        self.sigma = self.sigma0 * float(span.mean())
        self.cov = np.eye(dim)
        self.p_sigma = np.zeros(dim)
        self.p_c = np.zeros(dim)

        # Sample the gen=0 population around the mean; the initial hook reads .pop.
        self.pop = np.clip(
            self.mean + self.sigma * self.rng.standard_normal((lam, dim)),
            self.bounds[:, 0], self.bounds[:, 1])
        self.fitness = None

    def _penalty(self, gen: int) -> float:
        # Use DE's linear penalty schedule for comparable optimizer ablations.
        return self.penalty_min + (self.penalty_max - self.penalty_min) * (gen / self.max_gen)

    def evaluate(self, gen: int):
        """Evaluate the current population using f + penalty*violation."""
        penalty = self._penalty(gen)
        # One vectorized population call through the experiment batch path.
        f = np.asarray(self.fitness_function(self.pop), dtype=float)
        c = np.asarray(self.constraint(self.pop), dtype=float)
        self.fitness = f + penalty * c

    def _sample(self):
        """Sample lambda points from N(mean, sigma^2*C), then clip to box bounds."""
        # Factorize once per iteration. Floor eigenvalues above zero to avoid
        # invalid square roots when C becomes nearly singular in long runs.
        eigvals, B = np.linalg.eigh(self.cov)
        eigvals = np.maximum(eigvals, 1e-20)
        D = np.sqrt(eigvals)
        Z = self.rng.standard_normal((self.lam, self.dim))
        Y = (B * D) @ Z.T            # (dim, lambda), with BD*z stored by column.
        X = self.mean + self.sigma * Y.T
        X = np.clip(X, self.bounds[:, 0], self.bounds[:, 1])
        return X, B, D

    def run(self):
        # The initial hook samples labels and fits the surrogate, as in DE.
        if self.online_hook is not None:
            self.online_hook(self, 0)

        for gen in range(1, self.max_gen + 1):
            X, B, D = self._sample()
            self.pop = X
            self.evaluate(gen)

            # Select the top mu members in ascending penalized fitness.
            idx = np.argsort(self.fitness)[: self.mu]
            X_sel = X[idx]

            mean_old = self.mean
            # Use clipped points to keep the mean feasible; normalize steps by sigma.
            self.mean = self.w @ X_sel
            y_w = (self.mean - mean_old) / self.sigma

            # Conjugate path with anisotropy correction: C^(-1/2)y_w = B D^(-1) B^T y_w.
            c_inv_sqrt_y = B @ ((B.T @ y_w) / D)
            self.p_sigma = ((1 - self.c_sigma) * self.p_sigma
                            + np.sqrt(self.c_sigma * (2 - self.c_sigma) * self.mu_eff)
                            * c_inv_sqrt_y)

            # Suppress the p_c rank-one direction when the p_sigma norm is too large.
            denom = np.sqrt(1 - (1 - self.c_sigma) ** (2 * gen)) * self.chi_n
            h_sigma = float(np.linalg.norm(self.p_sigma) / denom
                            < 1.4 + 2 / (self.dim + 1))
            self.p_c = ((1 - self.c_c) * self.p_c
                        + h_sigma * np.sqrt(self.c_c * (2 - self.c_c) * self.mu_eff)
                        * y_w)

            # Rank-one (p_c) plus rank-mu (selected steps) covariance updates.
            Y_sel = (X_sel - mean_old) / self.sigma     # (mu, dim)
            self.cov = ((1 - self.c_1 - self.c_mu) * self.cov
                        + self.c_1 * np.outer(self.p_c, self.p_c)
                        + self.c_mu * (Y_sel * self.w[:, None]).T @ Y_sel)

            # Step-size adaptation.
            self.sigma *= float(np.exp(
                (self.c_sigma / self.d_sigma)
                * (np.linalg.norm(self.p_sigma) / self.chi_n - 1)))
            # Floor sigma to prevent long-run sampling from collapsing to a single point.
            self.sigma = max(self.sigma, 1e-16)

            # Trigger the online hook every ever_gen generations, as in DE.
            if self.online_hook is not None and (gen % self.ever_gen == 0):
                self.online_hook(self, gen)
                self.evaluate(gen)   # Reevaluate the current population after the surrogate changes.

        # Inject the CMA-ES mean into the worst population slot so final_real
        # diagnostics include the optimizer's final estimated solution.
        mean_clip = np.clip(self.mean, self.bounds[:, 0], self.bounds[:, 1])
        fit_mean = (float(self.fitness_function(mean_clip))
                    + self._penalty(self.max_gen)
                    * float(self.constraint(mean_clip)))
        worst = int(np.argmax(self.fitness))
        self.pop[worst] = mean_clip
        self.fitness[worst] = fit_mean

        return self.pop, self.fitness
