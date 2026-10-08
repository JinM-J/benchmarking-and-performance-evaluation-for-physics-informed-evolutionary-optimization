"""Black-box GLoSADE adaptation to the common benchmark budget.

Y. Wang, D.-Q. Yin, S. Yang, and G. Sun, "Global and Local
Surrogate-Assisted Differential Evolution for Expensive Constrained
Optimization Problems With Inequality Constraints," IEEE Transactions on
Cybernetics, 49(5), 1642--1656, 2019, doi:10.1109/TCYB.2018.2809430.

The global stage randomly splits the population between GRNN-guided
DE/rand/1/bin and cubic-RBF-uncertainty-guided DE/current-to-rand/1.
Each target selects one of lambda free candidates for a true evaluation,
then competes with its parent under Deb's feasibility rule. A local stage
uses KNN cubic RBF models and constrained optimization. Separate surrogate
outputs model the objective and explicit constraints; no PDE information
or state-field interface is accessed.

Report this version as GLoSADE-adapted. Equalities in F02/F08 are converted
to abs(h)-tol<=0 with the declared tolerance, while Problem.violation checks
the final state. The original NP=80, K>=100 and several thousand evaluations
are adapted to the configured HF budget: protocol.init_points sets the initial
population and K is capped by archive size. Lambda, DE parameters and GRNN
spread retain the paper/official-code values. SciPy trust-constr replaces
MATLAB fmincon(interior-point), with the same 300-iteration cap. This is not
a bitwise reproduction of the original experimental setup.
"""
from __future__ import annotations

import time
import warnings

import numpy as np
from scipy.optimize import Bounds, NonlinearConstraint, minimize

from methods.base import BenchmarkMethod
from surrogates.glosade_models import (
    CubicRBF,
    GLoSADEGRNN,
    cubic_rbf_uncertainty,
)


class GLoSADEDriver:
    """Self-managed GLoSADE loop; all true evaluations use DecisionEvaluator."""

    def __init__(self, problem, protocol, evaluator, seed: int):
        self.problem = problem
        self.protocol = protocol
        self.evaluator = evaluator
        self.rng = np.random.default_rng(seed)
        self.bounds = np.asarray(problem.decision_bounds, dtype=float)
        if (self.bounds.ndim != 2 or self.bounds.shape[1] != 2
                or not np.isfinite(self.bounds).all()
                or np.any(self.bounds[:, 1] <= self.bounds[:, 0])):
            raise ValueError("GLoSADE requires finite (d,2) decision_bounds with strictly positive spans")

        dim = self.bounds.shape[0]
        default_local_k = max((dim + 1) * (dim + 2) // 2, 100)
        mp = getattr(protocol, "method_params", {}) or {}
        self.init_points = int(protocol.init_points)
        self.n_candidates = int(mp.get("glosade_lambda", 100))
        self.grnn_spread = float(mp.get("glosade_grnn_spread", 0.5))
        self.f_rand = float(mp.get("glosade_f_rand", 0.8))
        self.crossover_rate = float(mp.get("glosade_cr", 0.4))
        self.f_current = float(mp.get("glosade_f_current", 0.4))
        self.uncertainty_k = int(mp.get("glosade_uncertainty_k", 100))
        self.local_k = int(mp.get("glosade_local_k", default_local_k))
        self.local_maxiter = int(mp.get("glosade_local_maxiter", 300))

        if self.init_points <= 0:
            raise ValueError("GLoSADE initialization requires protocol.init_points > 0")
        if self.n_candidates <= 0:
            raise ValueError("glosade_lambda must be a positive integer")
        if self.grnn_spread <= 0.0:
            raise ValueError("glosade_grnn_spread must be positive")
        if not (0.0 <= self.crossover_rate <= 1.0):
            raise ValueError("glosade_cr must lie in [0,1]")
        if self.f_rand < 0.0 or self.f_current < 0.0:
            raise ValueError("GLoSADE mutation factors must be nonnegative")
        if self.uncertainty_k <= 0 or self.local_k <= 0:
            raise ValueError("GLoSADE KNN neighbor counts must be positive integers")
        if self.local_maxiter <= 0:
            raise ValueError("glosade_local_maxiter must be a positive integer")

        self.history = []
        self.surrogate_train_calls = 0
        self.surrogate_train_time = 0.0
        self.surrogate_infer_time = 0.0

    # True evaluation budget and explicit constraints.
    def _spent(self) -> int:
        oracle = self.evaluator.oracle
        if getattr(self.protocol, "budget_kind", "state") == "parameter":
            return int(oracle.n_parameter_queries)
        return int(oracle.n_state_queries)

    def _remaining(self) -> int:
        return max(0, int(self.protocol.hf_budget) - self._spent())

    @staticmethod
    def _surrogate_constraints(C, cons_meta) -> np.ndarray:
        """Map raw g/h components to G<=0, retaining negative feasibility margins."""
        C = np.asarray(C, dtype=float)
        if C.ndim != 2:
            raise ValueError("Explicit constraint matrix must have shape (n,nc)")
        if C.shape[1] != len(cons_meta):
            raise RuntimeError(
                "constraint_components and constraint_meta channel counts differ: "
                f"received {C.shape[1]}, declared {len(cons_meta)}"
            )
        G = np.empty_like(C)
        for j, meta in enumerate(cons_meta):
            tol = float(meta["tol"])
            if meta["kind"] == "g":
                G[:, j] = C[:, j] - tol
            elif meta["kind"] == "h":
                G[:, j] = np.abs(C[:, j]) - tol
            else:
                raise ValueError(f"Unknown constraint type: {meta['kind']!r}")
        return G

    @staticmethod
    def _cv(G) -> np.ndarray:
        G = np.asarray(G, dtype=float)
        if G.ndim != 2:
            raise ValueError("Surrogate constraint matrix must have shape (n,nc)")
        if G.shape[1] == 0:
            return np.zeros(G.shape[0], dtype=float)
        return np.sum(np.maximum(G, 0.0), axis=1)

    @staticmethod
    def _better(f1, cv1, f2, cv2) -> bool:
        """Deb feasibility rule from PairSelect; replace only on strict improvement."""
        feasible1 = cv1 <= 0.0
        feasible2 = cv2 <= 0.0
        if feasible1 != feasible2:
            return bool(feasible1)
        if feasible1:
            return bool(f1 < f2)
        return bool(cv1 < cv2)

    def _best_index(self, f, G) -> int:
        f = np.asarray(f, dtype=float).reshape(-1)
        cv = self._cv(G)
        best = 0
        for i in range(1, len(f)):
            if self._better(f[i], cv[i], f[best], cv[best]):
                best = i
        return int(best)

    def _validate_real(self, X, f, C, cons_meta):
        X = np.atleast_2d(np.asarray(X, dtype=float))
        f = np.asarray(f, dtype=float).reshape(-1)
        C = np.asarray(C, dtype=float)
        if C.ndim == 1:
            C = C.reshape(len(X), -1)
        if X.shape[1] != self.bounds.shape[0]:
            raise ValueError("Invalid GLoSADE decision dimension for true evaluation")
        if np.any(X < self.bounds[:, 0]) or np.any(X > self.bounds[:, 1]):
            raise ValueError("GLoSADE true-evaluation decisions exceed decision_bounds")
        if f.shape != (len(X),) or C.shape != (len(X), len(cons_meta)):
            raise RuntimeError("GLoSADE true-evaluation output shape does not match the problem definition")
        if not np.isfinite(f).all() or not np.isfinite(C).all():
            raise RuntimeError("GLoSADE true evaluation returned nonfinite objective or constraint values")
        return X, f, C

    def _valid_decisions(self, X) -> np.ndarray:
        """Filter decisions using declared query-domain geometry, without PDE or field access."""
        X = np.atleast_2d(np.asarray(X, dtype=float))
        if X.shape[1] != self.bounds.shape[0]:
            raise ValueError("Invalid GLoSADE decision dimension for domain validation")
        Q = np.asarray(
            [self.problem.decision_to_query(x) for x in X], dtype=float
        )
        valid = np.asarray(self.problem.is_valid_query(Q), dtype=bool)
        if valid.shape != (len(X),):
            raise RuntimeError(
                "Problem.is_valid_query must return one boolean per GLoSADE decision"
            )
        return valid

    def _valid_latin_hypercube(self, n: int) -> np.ndarray:
        """Reject invalid LHS points and replenish to maintain the initial population size."""
        accepted = []
        need = int(n)
        for _ in range(100):
            candidates = self._latin_hypercube(need)
            valid = candidates[self._valid_decisions(candidates)]
            if len(valid):
                accepted.append(valid)
                need -= len(valid)
            if need == 0:
                return np.vstack(accepted)
        raise RuntimeError(
            f"GLoSADE still needs {need} valid initial points after 100 LHS replenishment rounds"
        )

    def _valid_trials(self, generator) -> np.ndarray:
        """Replenish free candidates so each target compares lambda valid points."""
        accepted = []
        need = self.n_candidates
        for _ in range(100):
            candidates = generator(need)
            valid = candidates[self._valid_decisions(candidates)]
            if len(valid):
                accepted.append(valid)
                need -= len(valid)
            if need == 0:
                return np.vstack(accepted)
        raise RuntimeError(
            f"GLoSADE still needs {need} valid points after 100 candidate replenishment rounds"
        )

    def _evaluate_one(self, x, arch_X, arch_f, arch_C, cons_meta):
        """Single entry point for true evaluations; also append to the global archive."""
        if self._remaining() <= 0:
            return None
        X = np.asarray(x, dtype=float).reshape(1, -1)
        if not self._valid_decisions(X)[0]:
            raise RuntimeError("GLoSADE cannot evaluate a decision outside the valid query domain")
        f, C, _ = self.evaluator.evaluate(X)
        X, f, C = self._validate_real(X, f, C, cons_meta)
        arch_X.append(X[0].copy())
        arch_f.append(float(f[0]))
        arch_C.append(C[0].copy())
        G = self._surrogate_constraints(C, cons_meta)
        return float(f[0]), C[0].copy(), G[0].copy()

    # Surrogate timing.
    def _fit(self, model, X, Y):
        t0 = time.perf_counter()
        model.fit(X, Y)
        self.surrogate_train_time += time.perf_counter() - t0
        self.surrogate_train_calls += 1
        return model

    def _predict(self, model, X):
        t0 = time.perf_counter()
        out = model.predict(X)
        self.surrogate_infer_time += time.perf_counter() - t0
        return out

    # Initialization and DE candidates.
    def _latin_hypercube(self, n: int) -> np.ndarray:
        dim = self.bounds.shape[0]
        unit = np.empty((n, dim), dtype=float)
        for j in range(dim):
            unit[:, j] = (self.rng.permutation(n) + self.rng.random(n)) / n
        return (self.bounds[:, 0]
                + unit * (self.bounds[:, 1] - self.bounds[:, 0]))

    def _repair(self, x) -> np.ndarray:
        """Official repair: randomly reflect or clip a candidate, then clip as a bounds guard."""
        x = np.asarray(x, dtype=float).copy()
        lo, hi = self.bounds[:, 0], self.bounds[:, 1]
        below, above = x < lo, x > hi
        if self.rng.random() <= 0.5:
            x[below] = np.minimum(hi[below], 2.0 * lo[below] - x[below])
            x[above] = np.maximum(lo[above], 2.0 * hi[above] - x[above])
        else:
            x[below] = lo[below]
            x[above] = hi[above]
        return np.clip(x, lo, hi)

    def _rand_1_bin(self, population, target: int,
                    n_candidates: int | None = None) -> np.ndarray:
        mu, dim = population.shape
        available = np.delete(np.arange(mu), target)
        n_candidates = (self.n_candidates if n_candidates is None
                        else int(n_candidates))
        U = np.empty((n_candidates, dim), dtype=float)
        for j in range(n_candidates):
            r1, r2, r3 = self.rng.choice(available, size=3, replace=False)
            mutant = (population[r1]
                      + self.f_rand * (population[r2] - population[r3]))
            cross = self.rng.random(dim) <= self.crossover_rate
            cross[int(self.rng.integers(dim))] = True
            U[j] = self._repair(np.where(cross, mutant, population[target]))
        return U

    def _current_to_rand_1(self, population, target: int,
                           n_candidates: int | None = None) -> np.ndarray:
        mu, dim = population.shape
        n_candidates = (self.n_candidates if n_candidates is None
                        else int(n_candidates))
        U = np.empty((n_candidates, dim), dtype=float)
        for j in range(n_candidates):
            # Official code allows r1==target; r2/r3 exclude both and each other.
            r1 = int(self.rng.integers(mu))
            excluded = {target, r1}
            available = np.array(
                [i for i in range(mu) if i not in excluded], dtype=int
            )
            r2, r3 = self.rng.choice(available, size=2, replace=False)
            mutant = (
                population[target]
                + self.f_current * (population[r1] - population[target])
                + self.f_current * (population[r2] - population[r3])
            )
            U[j] = self._repair(mutant)
        return U

    def _global_candidates(self, population, arch_X, arch_f, arch_G):
        X = np.asarray(arch_X, dtype=float)
        Y = np.column_stack([np.asarray(arch_f, dtype=float),
                             np.asarray(arch_G, dtype=float)])
        grnn = self._fit(GLoSADEGRNN(self.grnn_spread), X, Y)

        mu = len(population)
        permutation = self.rng.permutation(mu)
        n_fitness = (mu + 1) // 2  # MATLAB round(mu/2) for positive integers.
        fitness_targets = set(permutation[:n_fitness].tolist())
        candidates = np.empty_like(population)
        for target in range(mu):
            if target in fitness_targets:
                trials = self._valid_trials(
                    lambda n: self._rand_1_bin(population, target, n)
                )
                pred = self._predict(grnn, trials)
                selected = self._best_index(pred[:, 0], pred[:, 1:])
            else:
                trials = self._valid_trials(
                    lambda n: self._current_to_rand_1(population, target, n)
                )
                t0 = time.perf_counter()
                uncertainty = cubic_rbf_uncertainty(
                    trials, X, self.uncertainty_k
                )
                self.surrogate_infer_time += time.perf_counter() - t0
                selected = int(np.argmax(uncertainty))
            candidates[target] = trials[selected]
        return candidates

    # Local cubic RBF models and constrained interior-point optimization.
    def _local_candidate(self, target, arch_X, arch_f, arch_G):
        X = np.asarray(arch_X, dtype=float)
        f = np.asarray(arch_f, dtype=float)
        G = np.asarray(arch_G, dtype=float)
        distance = np.linalg.norm(X - target, axis=1)
        ordered = np.argsort(distance, kind="stable")

        # Official code requests K+1 neighbors and removes the first (the parent).
        # Remove only one row even when the archive contains duplicate points.
        k_eff = min(self.local_k, max(0, len(X) - 1))
        if k_eff == 0:
            return np.asarray(target, dtype=float).copy()
        near = ordered[1:k_eff + 1]
        Yn = np.column_stack([f[near], G[near]])
        model = self._fit(CubicRBF(), X[near], Yn)

        local_lo = np.min(np.vstack([target, X[near]]), axis=0)
        local_hi = np.max(np.vstack([target, X[near]]), axis=0)
        width = local_hi - local_lo
        narrow = width <= 2e-8
        local_lo[narrow] -= (2e-8 - width[narrow]) / 2.0
        local_hi[narrow] += (2e-8 - width[narrow]) / 2.0
        local_lo = np.maximum(local_lo, self.bounds[:, 0])
        local_hi = np.minimum(local_hi, self.bounds[:, 1])

        # Numerical guard for constant axes.
        still_narrow = local_hi <= local_lo
        if np.any(still_narrow):
            span = self.bounds[:, 1] - self.bounds[:, 0]
            eps = np.minimum(1e-8, 0.5 * span)
            local_lo[still_narrow] = np.maximum(
                self.bounds[still_narrow, 0], target[still_narrow] - eps[still_narrow]
            )
            local_hi[still_narrow] = np.minimum(
                self.bounds[still_narrow, 1], target[still_narrow] + eps[still_narrow]
            )

        def predict_one(z):
            return self._predict(model, np.asarray(z).reshape(1, -1))[0]

        def objective(z):
            return float(predict_one(z)[0])

        constraints = ()
        if G.shape[1]:
            constraints = (NonlinearConstraint(
                lambda z: predict_one(z)[1:],
                -np.inf * np.ones(G.shape[1]),
                np.zeros(G.shape[1]),
            ),)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="delta_grad == 0.0", category=UserWarning
            )
            result = minimize(
                objective,
                np.clip(target, local_lo, local_hi),
                method="trust-constr",
                bounds=Bounds(local_lo, local_hi),
                constraints=constraints,
                options={"maxiter": self.local_maxiter, "verbose": 0},
            )
        candidate = np.asarray(result.x, dtype=float)
        if candidate.shape != target.shape or not np.isfinite(candidate).all():
            return np.asarray(target, dtype=float).copy()
        candidate = np.clip(candidate, self.bounds[:, 0], self.bounds[:, 1])
        if not self._valid_decisions(candidate.reshape(1, -1))[0]:
            # trust-constr cannot represent holes; the parent is already a valid query.
            return np.asarray(target, dtype=float).copy()
        return candidate

    # State updates and main loop.
    def _record(self, cycle, phase, population, pop_f, pop_G, n_archive):
        cv = self._cv(pop_G)
        feasible = cv <= 0.0
        self.history.append({
            "gen": int(cycle),
            "eval_count": self._spent(),
            "mse200": float("nan"),
            "real_best": (float(np.min(np.asarray(pop_f)[feasible]))
                          if feasible.any() else float("nan")),
            "feasible_ratio": float(np.mean(feasible)),
            "n_archive": int(n_archive),
            "phase": phase,
        })

    def _evaluate_phase(self, candidates, population, pop_f, pop_C, pop_G,
                        arch_X, arch_f, arch_C, arch_G, cons_meta):
        n_evaluated = 0
        for target, candidate in enumerate(candidates):
            if self._remaining() <= 0:
                break
            value = self._evaluate_one(
                candidate, arch_X, arch_f, arch_C, cons_meta
            )
            if value is None:
                break
            f_new, C_new, G_new = value
            arch_G.append(G_new.copy())
            cv_new = float(self._cv(G_new.reshape(1, -1))[0])
            cv_old = float(self._cv(pop_G[target].reshape(1, -1))[0])
            if self._better(f_new, cv_new, pop_f[target], cv_old):
                population[target] = candidate
                pop_f[target] = f_new
                pop_C[target] = C_new
                pop_G[target] = G_new
            n_evaluated += 1
        return n_evaluated

    def run(self):
        budget = int(self.protocol.hf_budget)
        n_init = min(self.init_points, budget)
        if n_init < 4:
            raise ValueError(
                "GLoSADE DE/rand/1 and current-to-rand/1 require at least 4 "
                f"true initial individuals; received {n_init}"
            )

        cons_meta = list(self.problem._glosade_cons_meta)
        population = self._valid_latin_hypercube(n_init)
        pop_f, pop_C, _ = self.evaluator.evaluate(population)
        population, pop_f, pop_C = self._validate_real(
            population, pop_f, pop_C, cons_meta
        )
        pop_G = self._surrogate_constraints(pop_C, cons_meta)
        arch_X = [x.copy() for x in population]
        arch_f = pop_f.astype(float).tolist()
        arch_C = [c.copy() for c in pop_C]
        arch_G = [g.copy() for g in pop_G]
        self._record(0, "init", population, pop_f, pop_G, len(arch_X))

        cycle = 0
        while self._remaining() > 0:
            cycle += 1
            spent_before = self._spent()

            global_candidates = self._global_candidates(
                population, arch_X, arch_f, arch_G
            )
            self._evaluate_phase(
                global_candidates, population, pop_f, pop_C, pop_G,
                arch_X, arch_f, arch_C, arch_G, cons_meta,
            )
            self._record(
                cycle, "global", population, pop_f, pop_G, len(arch_X)
            )

            if self._remaining() > 0:
                # Construct all local candidates from the same archive snapshot at phase start.
                # Then evaluate them together, matching the official MATLAB DB snapshot semantics.
                local_candidates = np.asarray([
                    self._local_candidate(x, arch_X, arch_f, arch_G)
                    for x in population
                ])
                self._evaluate_phase(
                    local_candidates, population, pop_f, pop_C, pop_G,
                    arch_X, arch_f, arch_C, arch_G, cons_meta,
                )
                self._record(
                    cycle, "local", population, pop_f, pop_G, len(arch_X)
                )

            if self._spent() <= spent_before:
                raise RuntimeError(
                    "GLoSADE true-evaluation phase did not increase protocol FE; stopped to prevent an infinite loop"
                )

        # Expose the full true-evaluation archive. Aggregate archive_vio using the same
        # constraint_meta/tol as PairSelect feasibility decisions.
        self.archive_dec = np.asarray(arch_X, dtype=float)
        self.archive_obj = np.asarray(arch_f, dtype=float)
        self.archive_vio = self._cv(np.asarray(arch_G, dtype=float))
        return population.copy(), pop_f.copy()


class GLoSADEMethod(BenchmarkMethod):
    """Wang et al. GLoSADE adapted to a common small-budget black-box protocol."""

    name = "GLoSADE-adapted"
    needs_decision_evaluator = True

    def __init__(self, protocol, seed: int):
        self.protocol = protocol
        self.seed = seed
        self.constraint_mode = "feasibility"
        self.surrogate = None
        self.surrogate_config = {}
        self.update_policy = None
        self.penalty = None
        self._evaluator = None

    def bind_evaluator(self, evaluator):
        self._evaluator = evaluator

    def setup(self, problem):
        # Read explicit decision constraints only; never access physics.
        problem._glosade_cons_meta = problem.constraint_meta()

    def build_optimizer(self, problem, protocol, fitness_fn, constraint_fn,
                        online_hook):
        if self._evaluator is None:
            raise RuntimeError("GLoSADEMethod requires a DecisionEvaluator supplied by Experiment")
        return GLoSADEDriver(problem, protocol, self._evaluator, self.seed)
