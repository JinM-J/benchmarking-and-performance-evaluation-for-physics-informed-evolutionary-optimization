"""Black-box Ji et al. SaDE-SA-GRM adapted to the common benchmark protocol.

Inputs are the declared decision vectors.
Models learn only scalar objectives and explicit constraints from true
evaluations; they do not access PDEs, state fields, derivatives, residuals,
initial/boundary conditions, collocation points or the physics interface.
All true evaluations, including GRM finite differences, use DecisionEvaluator
and are charged by HighFidelityOracle under the selected protocol.
protocol.init_points determines the true initial sample count, while
protocol.pop_size determines the free surrogate-search population.

The method combines objective/per-constraint cubic RBFNs, five-strategy
SaDE, Algorithm 2 constraint-wise consensus selection, local RBFN search after
stagnation, and GRM with true finite-difference gradients. This does not claim
bitwise reproduction of the original initialization or experiments; report as
"Ji et al. SaDE-SA-GRM adapted to the unified benchmark protocol".
"""
import numpy as np

from methods.base import BenchmarkMethod
from optimizers.sade_de import SADE_DE
from surrogates.rbfn import JiSurrogateBundle


class JiSADEDriver:
    """Self-managed Ji loop; run() returns the true-evaluation archive and objectives."""

    def __init__(self, problem, protocol, evaluator, seed: int):
        self.problem = problem
        self.protocol = protocol
        self.evaluator = evaluator
        self.rng = np.random.default_rng(seed)
        self.bounds = np.asarray(problem.decision_bounds, dtype=float)
        if self.bounds.ndim != 2 or self.bounds.shape[1] != 2:
            raise ValueError("Ji requires decision_bounds with shape (d, 2)")

        mp = getattr(protocol, "method_params", {}) or {}
        self.pop_size = int(protocol.pop_size)
        self.init_points = int(protocol.init_points)
        self.inner_gens = int(mp.get("inner_gens", 200))
        self.stagnation_limit = int(mp.get("stagnation_limit", 5))
        self.fd_rel_step = float(mp.get("grm_fd_rel_step", 1e-4))
        if self.pop_size < 6:
            raise ValueError("Ji SaDE requires protocol.pop_size >= 6")
        if self.init_points <= 0:
            raise ValueError("Ji initialization requires protocol.init_points > 0")
        if self.inner_gens <= 0 or self.stagnation_limit <= 0:
            raise ValueError("inner_gens and stagnation_limit must be positive integers")
        if not (0.0 < self.fd_rel_step < 1.0):
            raise ValueError("grm_fd_rel_step must lie in (0, 1)")

        self.history = []

    # True evaluation budget and constraint semantics.
    def _spent(self) -> int:
        oracle = self.evaluator.oracle
        if getattr(self.protocol, "budget_kind", "state") == "parameter":
            return int(oracle.n_parameter_queries)
        return int(oracle.n_state_queries)

    def _remaining(self) -> int:
        return max(0, int(self.protocol.hf_budget) - self._spent())

    @staticmethod
    def _constraint_violations(C, cons_meta) -> np.ndarray:
        C = np.asarray(C, dtype=float)
        if C.ndim != 2:
            raise ValueError("Constraint matrix must have shape (n, nc)")
        if C.shape[1] == 0:
            return C.copy()
        V = np.empty_like(C)
        for j, meta in enumerate(cons_meta):
            if meta["kind"] == "g":
                V[:, j] = np.maximum(C[:, j] - meta["tol"], 0.0)
            elif meta["kind"] == "h":
                V[:, j] = np.maximum(np.abs(C[:, j]) - meta["tol"], 0.0)
            else:
                raise ValueError(f"Unknown constraint type: {meta['kind']!r}")
        return V

    def _cv_true(self, C, cons_meta) -> np.ndarray:
        V = self._constraint_violations(C, cons_meta)
        return np.sum(V, axis=1) if V.shape[1] else np.zeros(V.shape[0])

    @staticmethod
    def _better(f1, cv1, f2, cv2) -> bool:
        """Ji feasibility rule; return True only on strict improvement."""
        feas1 = cv1 <= 0.0
        feas2 = cv2 <= 0.0
        if feas1 != feas2:
            return bool(feas1)
        if feas1:
            return bool(f1 < f2)
        return bool(cv1 < cv2)

    def _best_index(self, f, cv) -> int:
        best = 0
        for j in range(1, len(f)):
            if self._better(f[j], cv[j], f[best], cv[best]):
                best = j
        return int(best)

    def _worst_index(self, f, cv) -> int:
        worst = 0
        for j in range(1, len(f)):
            if self._better(f[worst], cv[worst], f[j], cv[j]):
                worst = j
        return int(worst)

    # Archive and true evaluations.
    @staticmethod
    def _archive_arrays(arch_X, arch_f, arch_C):
        X = np.asarray(arch_X, dtype=float)
        f = np.asarray(arch_f, dtype=float)
        C = np.asarray(arch_C, dtype=float)
        if C.ndim == 1:
            C = C.reshape(len(X), -1)
        return X, f, C

    def _evaluate_append(self, X, arch_X, arch_f, arch_C) -> int:
        """Evaluate and append to the archive; return the number of evaluated decisions."""
        X = np.atleast_2d(np.asarray(X, dtype=float))
        if X.shape[1] != self.bounds.shape[0]:
            raise ValueError(
                f"Ji decision dimension must be {self.bounds.shape[0]}; received {X.shape[1]}"
            )
        if (np.any(X < self.bounds[:, 0])
                or np.any(X > self.bounds[:, 1])):
            raise ValueError("Ji true-evaluation decisions exceed decision_bounds")
        n_take = min(X.shape[0], self._remaining())
        X = X[:n_take]
        if X.shape[0] == 0:
            return 0
        f, C, _ = self.evaluator.evaluate(X)
        expected_meta = getattr(getattr(self, "problem", None),
                                "_ji_cons_meta", None)
        if expected_meta is not None and C.shape != (len(X), len(expected_meta)):
            raise RuntimeError(
                "constraint_components and constraint_meta channel counts differ: "
                f"received {C.shape}; declared shape is ({len(X)}, {len(expected_meta)})"
            )
        if not np.isfinite(f).all() or not np.isfinite(C).all():
            raise RuntimeError("Ji true evaluation returned nonfinite objective or constraint values")
        arch_X.extend(X.copy())
        arch_f.extend(np.asarray(f, dtype=float).tolist())
        arch_C.extend(np.asarray(C, dtype=float).copy())
        return int(X.shape[0])

    def _is_novel(self, x, archive_X) -> bool:
        if not archive_X:
            return True
        A = np.asarray(archive_X, dtype=float)
        span = self.bounds[:, 1] - self.bounds[:, 0]
        dist = np.max(np.abs((A - np.asarray(x, dtype=float)) / span), axis=1)
        return bool(np.all(dist > 1e-10))

    def _record(self, cycle, phase, arch_X, arch_f, arch_C, cons_meta):
        X, f, C = self._archive_arrays(arch_X, arch_f, arch_C)
        cv = self._cv_true(C, cons_meta)
        feasible = cv <= 0.0
        self.history.append({
            "gen": int(cycle),
            "eval_count": self._spent(),
            "mse200": float("nan"),
            "real_best": (float(np.min(f[feasible]))
                          if feasible.any() else float("nan")),
            "feasible_ratio": float(np.mean(feasible)),
            "n_archive": int(len(X)),
            "phase": phase,
        })

    # Algorithm 2: constraint-wise consensus selection.
    def _consensus_order(self, P, Q, bundle) -> np.ndarray:
        """Rank Q by per-channel consensus, falling back to CV consensus."""
        Vp = bundle.predict_constraint_violations(P)
        Vq = bundle.predict_constraint_violations(Q)
        fq = bundle.predict_f(Q)
        cvq = np.sum(Vq, axis=1) if Vq.shape[1] else np.zeros(len(Q))

        if Vq.shape[1] == 0:
            eligible = np.arange(len(Q))
        else:
            mask = np.all(Vq <= Vp, axis=1)
            if not mask.any():
                mask = cvq <= np.sum(Vp, axis=1)
            eligible = np.where(mask)[0]

        if eligible.size:
            first = eligible[np.argsort(fq[eligible], kind="stable")]
        else:
            first = np.empty(0, dtype=int)
        used = np.zeros(len(Q), dtype=bool)
        used[first] = True
        rest = np.where(~used)[0]
        if rest.size:
            rest = rest[np.lexsort((fq[rest], cvq[rest]))]
        return np.concatenate([first, rest]).astype(int)

    def _first_novel(self, candidates, order, archive_X):
        for idx in np.asarray(order, dtype=int):
            if self._is_novel(candidates[idx], archive_X):
                return np.asarray(candidates[idx], dtype=float).copy()
        return None

    def _insert_population(self, population, x, bundle):
        """Insert true infill into the free population under the surrogate feasibility rule."""
        fp = bundle.predict_f(population)
        cvp = bundle.predict_cv(population)
        fx = float(bundle.predict_f(np.asarray(x).reshape(1, -1))[0])
        cvx = float(bundle.predict_cv(np.asarray(x).reshape(1, -1))[0])
        worst = self._worst_index(fp, cvp)
        if self._better(fx, cvx, fp[worst], cvp[worst]):
            population[worst] = x

    # Algorithm 4: local RBFN and local SaDE.
    def _local_candidate(self, arch_X, arch_f, arch_C, cons_meta):
        X, f, C = self._archive_arrays(arch_X, arch_f, arch_C)
        cv = self._cv_true(C, cons_meta)
        best = self._best_index(f, cv)
        span = self.bounds[:, 1] - self.bounds[:, 0]
        dist = np.linalg.norm((X - X[best]) / span, axis=1)
        n_train = min(len(X), max(self.bounds.shape[0] + 1, self.pop_size))
        near = np.argsort(dist, kind="stable")[:n_train]
        Xn, fn, Cn = X[near], f[near], C[near]

        local_lo = np.min(Xn, axis=0)
        local_hi = np.max(Xn, axis=0)
        floor = 1e-6 * span
        narrow = local_hi - local_lo < floor
        local_lo[narrow] = np.maximum(
            self.bounds[narrow, 0], X[best, narrow] - 0.5 * floor[narrow]
        )
        local_hi[narrow] = np.minimum(
            self.bounds[narrow, 1], X[best, narrow] + 0.5 * floor[narrow]
        )
        local_bounds = np.column_stack([local_lo, local_hi])

        # Fit on neighboring archive points; use global bounds to avoid constant local axes.
        local_bundle = JiSurrogateBundle(self.bounds, cons_meta)
        local_bundle.fit(Xn, fn, Cn)
        initial = self.rng.uniform(local_lo, local_hi,
                                   (self.pop_size, self.bounds.shape[0]))
        n_inject = min(self.pop_size, len(Xn))
        initial[:n_inject] = Xn[:n_inject]
        de = SADE_DE(
            local_bounds,
            self.pop_size,
            evaluate_fn=lambda D: (local_bundle.predict_f(D),
                                   local_bundle.predict_cv(D)),
            seed=int(self.rng.integers(0, 2**31 - 1)),
            initial_pop=initial,
        )
        de.evolve(self.inner_gens)
        Q = de.candidates()
        order = self._rank_candidates(local_bundle.predict_f(Q),
                                      local_bundle.predict_cv(Q))
        return self._first_novel(Q, order, arch_X)

    @staticmethod
    def _rank_candidates(f, cv) -> np.ndarray:
        feasible = cv <= 0.0
        first = np.where(feasible)[0]
        first = first[np.argsort(f[first], kind="stable")]
        rest = np.where(~feasible)[0]
        rest = rest[np.lexsort((f[rest], cv[rest]))]
        return np.concatenate([first, rest]).astype(int)

    # Algorithm 3: GRM with true finite-difference evaluations.
    @staticmethod
    def _repair_residual(c, cons_meta):
        """Map raw g/h to signed residuals that vanish at the tolerance boundary."""
        residual = np.zeros(len(cons_meta), dtype=float)
        for j, meta in enumerate(cons_meta):
            tol = float(meta["tol"])
            if meta["kind"] == "g":
                residual[j] = max(c[j] - tol, 0.0)
            elif c[j] > tol:
                residual[j] = c[j] - tol
            elif c[j] < -tol:
                residual[j] = c[j] + tol
        return residual

    def _run_true_grm(self, start_index, arch_X, arch_f, arch_C,
                      cons_meta) -> bool:
        """Estimate gradients from d true finite-difference points and reuse them in GRM.

        Return whether GRM reduces the initial CV. All gradient and trial points
        enter the archive; no surrogate gradients or PDE residuals are used.
        """
        if not cons_meta:
            return False
        dim = self.bounds.shape[0]
        if self._remaining() < dim + 1:
            return False

        x = np.asarray(arch_X[start_index], dtype=float).copy()
        c = np.asarray(arch_C[start_index], dtype=float).copy()
        cv = float(self._cv_true(c.reshape(1, -1), cons_meta)[0])
        start_cv = cv
        if cv <= 0.0:
            return False

        span = self.bounds[:, 1] - self.bounds[:, 0]
        X_fd = np.repeat(x.reshape(1, -1), dim, axis=0)
        delta = np.empty(dim, dtype=float)
        for j in range(dim):
            step = self.fd_rel_step * span[j]
            if x[j] + step <= self.bounds[j, 1]:
                X_fd[j, j] = x[j] + step
            else:
                X_fd[j, j] = x[j] - step
            delta[j] = X_fd[j, j] - x[j]

        before = len(arch_X)
        if self._evaluate_append(X_fd, arch_X, arch_f, arch_C) != dim:
            return False
        C_fd = np.asarray(arch_C[before:before + dim], dtype=float)
        gradient = ((C_fd - c.reshape(1, -1)) / delta.reshape(-1, 1)).T

        # Reuse the same true gradient; cap iterations to avoid loops when cached parameters cost no FE.
        for _ in range(max(1, int(self.protocol.hf_budget))):
            if self._remaining() <= 0:
                break
            residual = self._repair_residual(c, cons_meta)
            violated = residual != 0.0
            if not violated.any():
                break
            step = np.linalg.pinv(gradient[violated]) @ residual[violated]
            x_trial = np.clip(x - step, self.bounds[:, 0], self.bounds[:, 1])
            if np.allclose(x_trial, x, rtol=0.0, atol=1e-14 * np.max(span)):
                break
            before = len(arch_X)
            if self._evaluate_append(x_trial, arch_X, arch_f, arch_C) != 1:
                break
            c_trial = np.asarray(arch_C[before], dtype=float)
            cv_trial = float(self._cv_true(c_trial.reshape(1, -1), cons_meta)[0])
            if cv_trial >= cv - 1e-12:
                break
            x, c, cv = x_trial, c_trial, cv_trial
            if cv <= 0.0:
                break
        return bool(cv < start_cv - 1e-12)

    # Main loop.
    def run(self):
        budget = int(self.protocol.hf_budget)
        dim = self.bounds.shape[0]
        n_init = min(self.init_points, budget)
        if n_init < dim + 1:
            raise ValueError(
                f"Ji cubic RBFN in {dim} dimensions requires init_points >= {dim + 1}; "
                f"received {n_init} valid initial points"
            )

        cons_meta = list(self.problem._ji_cons_meta)
        X0 = self.rng.uniform(self.bounds[:, 0], self.bounds[:, 1],
                              (n_init, dim))
        arch_X, arch_f, arch_C = [], [], []
        self._evaluate_append(X0, arch_X, arch_f, arch_C)
        self._record(0, "init", arch_X, arch_f, arch_C, cons_meta)

        # Use protocol.pop_size for free search, inserting true initial points before random fill.
        population = self.rng.uniform(self.bounds[:, 0], self.bounds[:, 1],
                                      (self.pop_size, dim))
        population[:min(n_init, self.pop_size)] = X0[:min(n_init, self.pop_size)]
        stagnation = 0
        cycle = 0
        no_spend_cycles = 0

        while self._remaining() > 0:
            cycle += 1
            X, f, C = self._archive_arrays(arch_X, arch_f, arch_C)
            bundle = JiSurrogateBundle(self.bounds, cons_meta)
            bundle.fit(X, f, C)
            cv_before = self._cv_true(C, cons_meta)
            incumbent_before = self._best_index(f, cv_before)

            # Global surrogate evolution and Algorithm 2 select one expensive infill point.
            base_population = population.copy()
            de = SADE_DE(
                self.bounds,
                self.pop_size,
                evaluate_fn=lambda D: (bundle.predict_f(D), bundle.predict_cv(D)),
                seed=int(self.rng.integers(0, 2**31 - 1)),
                initial_pop=base_population,
            )
            de.evolve(self.inner_gens)
            evolved = de.candidates()
            order = self._consensus_order(base_population, evolved, bundle)
            candidate = self._first_novel(evolved, order, arch_X)
            if candidate is None:
                candidate = self.rng.uniform(self.bounds[:, 0], self.bounds[:, 1])

            spent_before = self._spent()
            before = len(arch_X)
            if self._evaluate_append(candidate, arch_X, arch_f, arch_C) != 1:
                break
            C_new = np.asarray(arch_C[before], dtype=float).reshape(1, -1)
            cv_new = float(self._cv_true(C_new, cons_meta)[0])
            improved = self._better(
                arch_f[before], cv_new,
                f[incumbent_before], cv_before[incumbent_before],
            )
            stagnation = 0 if improved else stagnation + 1
            population = base_population
            self._insert_population(population, candidate, bundle)
            phase = "global"

            # After snum=5 true infills without improvement, run local surrogate evolution.
            # If still unimproved and no true feasible point exists, invoke true FE-GRM.
            if stagnation >= self.stagnation_limit and self._remaining() > 0:
                phase = "local"
                X1, f1, C1 = self._archive_arrays(arch_X, arch_f, arch_C)
                cv1 = self._cv_true(C1, cons_meta)
                incumbent1 = self._best_index(f1, cv1)
                local = self._local_candidate(arch_X, arch_f, arch_C, cons_meta)
                local_improved = False
                if local is not None and self._remaining() > 0:
                    local_at = len(arch_X)
                    self._evaluate_append(local, arch_X, arch_f, arch_C)
                    C_local = np.asarray(arch_C[local_at], dtype=float).reshape(1, -1)
                    cv_local = float(self._cv_true(C_local, cons_meta)[0])
                    local_improved = self._better(
                        arch_f[local_at], cv_local,
                        f1[incumbent1], cv1[incumbent1],
                    )

                X2, f2, C2 = self._archive_arrays(arch_X, arch_f, arch_C)
                cv2 = self._cv_true(C2, cons_meta)
                if (not local_improved and not np.any(cv2 <= 0.0)
                        and cons_meta and self._remaining() >= dim + 1):
                    phase = "local+grm"
                    start = self._best_index(f2, cv2)
                    self._run_true_grm(start, arch_X, arch_f, arch_C, cons_meta)
                stagnation = 0

            self._record(cycle, phase, arch_X, arch_f, arch_C, cons_meta)
            no_spend_cycles = (no_spend_cycles + 1
                               if self._spent() == spent_before else 0)
            if no_spend_cycles >= max(10, self.pop_size):
                raise RuntimeError("Ji consecutive true evaluations did not increase protocol FE; stopped to prevent an infinite loop")

        X, f, C = self._archive_arrays(arch_X, arch_f, arch_C)
        self.archive_dec = X.copy()
        self.archive_obj = f.copy()
        self.archive_vio = self._cv_true(C, cons_meta)
        return X, f


class JiSADEGRMMethod(BenchmarkMethod):
    """Ji et al. black-box SAEA driven by the common decision-level true evaluator."""

    name = "Ji-SaDE-SA-GRM-adapted"
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
        # Read explicit decision constraints only; do not access problem.physics.
        problem._ji_cons_meta = problem.constraint_meta()

    def build_optimizer(self, problem, protocol, fitness_fn, constraint_fn,
                        online_hook):
        if self._evaluator is None:
            raise RuntimeError("JiSADEGRMMethod requires a DecisionEvaluator supplied by Experiment")
        return JiSADEDriver(problem, protocol, self._evaluator, self.seed)
