# core/experiment.py
"""Run one experiment by composing a Problem, Method, and Protocol.

Use interfaces rather than concrete method types. Selection is delegated to
method.update_policy (HFRequest); the oracle owns FE accounting. Problems
define reduced objective/constraint evaluation, including quadrature.
The training pool stores dec/Q/u/f/vio as float32; f/vio are computed in
float64 before storage. Snapshot fields retain the reference-run schema."""
from dataclasses import dataclass

import numpy as np

from evaluation.oracle import HighFidelityOracle
from evaluation.metrics import MetricGrid
from evaluation.metrics.constraint import FEASIBLE_TOL


class _ReferenceProvider:
    """Read u directly from the reference dataset for diagnostic evaluation.

    Final-population and snapshot diagnostics bypass the oracle and consume
    neither FE nor RNG draws. Labels retain reference precision; float32
    rounding applies to training-pool storage only."""

    def __init__(self, reference):
        self.reference = reference

    def evaluate(self, query_points: np.ndarray) -> np.ndarray:
        return np.asarray(self.reference.query(query_points),
                          dtype=float).reshape(-1)


class _SurrogateProvider:
    """Predict u with the surrogate for optimizer fitness evaluation.

    query_dtype controls input rounding (e.g. float32 for F09). None retains
    float64 queries."""

    def __init__(self, surrogate, query_dtype=None):
        self.surrogate = surrogate
        self.query_dtype = query_dtype
        self.total_time = 0.0   # Cumulative surrogate inference time.
        self._batch_q = None    # Pending batch queries; None selects ordinary evaluation.
        self._batch_cache = None  # Query-row bytes -> u, reused for second-pass objective evaluation.

    def begin_batch(self):
        """Collect queries and return placeholder zeros; discard first-pass algebra."""
        self._batch_q = []

    def flush_batch(self):
        """Predict collected queries once and cache each row for second-pass algebra.

        Batch and pointwise sklearn predictions can differ by about one ULP;
        the batch path is deterministic for identical batches."""
        import time
        if not self._batch_q:
            # Unconstrained violation() may never query the state provider.
            # Use the first-pass return value in that case instead of stacking an empty list.
            self._batch_q = None
            self._batch_cache = None
            return False
        t0 = time.perf_counter()
        u_all = self.surrogate.predict(np.vstack(self._batch_q))
        self.total_time += time.perf_counter() - t0
        self._batch_cache = {}
        i0 = 0
        for Q in self._batch_q:
            for i in range(Q.shape[0]):
                self._batch_cache[Q[i].tobytes()] = float(u_all[i0 + i])
            i0 += Q.shape[0]
        self._batch_q = None
        return True

    def clear_batch(self):
        """Discard pending queries and cached values, restoring pointwise evaluation."""
        self._batch_q = None
        self._batch_cache = None

    def evaluate(self, query_points: np.ndarray) -> np.ndarray:
        Q = np.asarray(query_points, dtype=float)
        if Q.ndim == 1:
            Q = Q.reshape(1, -1)
        if self.query_dtype is not None:
            Q = Q.astype(self.query_dtype)
        if self._batch_q is not None:
            # Collect queries only; placeholder values invalidate first-pass algebra.
            self._batch_q.append(Q)
            return np.zeros(Q.shape[0], dtype=float)
        if self._batch_cache is not None:
            # Serve cached batch predictions during the second pass.
            return np.asarray([self._batch_cache[Q[i].tobytes()]
                               for i in range(Q.shape[0])], dtype=float)
        import time
        t0 = time.perf_counter()
        out = self.surrogate.predict(Q)
        self.total_time += time.perf_counter() - t0
        return out


class _PointProvider:
    """Use anchored sample labels, falling back to reference data elsewhere.

    Pointwise objectives (F01-F08) query only the anchor. Integral objectives
    (F09+) use reference interpolation at quadrature points as diagnostic
    evaluation, bypassing the oracle without consuming FE."""

    def __init__(self, query: np.ndarray, u: float, reference):
        self.anchor = np.asarray(query, dtype=float).reshape(-1)
        self.u = float(u)
        self.reference = reference

    def evaluate(self, query_points: np.ndarray) -> np.ndarray:
        Q = np.asarray(query_points, dtype=float)
        if Q.ndim == 1:
            Q = Q.reshape(1, -1)
        if Q.shape[0] >= 1 and np.array_equal(Q[0], self.anchor):
            if Q.shape[0] == 1:
                return np.array([self.u], dtype=float)
            rest = self.reference.query(Q[1:])
            return np.concatenate([[self.u], np.asarray(rest, dtype=float).reshape(-1)])
        return np.asarray(self.reference.query(Q), dtype=float).reshape(-1)


class _BlockProvider:
    """Anchor a full batch of labels for blockwise pool objectives.

    Reconstructing J(theta) requires all surface points for the same theta.
    The sampling policy includes that block in the batch, so reconstruction
    needs no additional solves. Match rows using keys rounded to 12 decimals;
    missing rows fall back to the reference provider."""

    def __init__(self, queries: np.ndarray, u: np.ndarray, reference):
        self.reference = reference
        self._map = {tuple(np.round(q, 12)): float(uu)
                     for q, uu in zip(np.asarray(queries, dtype=float), u)}

    def evaluate(self, query_points: np.ndarray) -> np.ndarray:
        Q = np.atleast_2d(np.asarray(query_points, dtype=float))
        out = np.empty(Q.shape[0], dtype=float)
        miss = []
        for i, q in enumerate(Q):
            v = self._map.get(tuple(np.round(q, 12)))
            if v is None:
                miss.append(i)
            else:
                out[i] = v
        if miss:
            out[miss] = np.asarray(self.reference.query(Q[miss]),
                                   dtype=float).reshape(-1)
        return out


@dataclass
class ExperimentResult:
    pop: np.ndarray
    fitness: np.ndarray
    db_history: list
    n_hf_queries: int
    n_state_queries: int
    n_parameter_queries: int
    hf_wall_time: float
    surrogate_train_calls: int
    # Timing breakdown: training, inference, HF evaluation, and total runtime.
    surrogate_train_time: float
    surrogate_infer_time: float
    # Reference diagnostics bypass the oracle and do not consume FE.
    real_watch_gen: list          # Snapshot generation indices.
    real_obj_best: list           # Best feasible true objective in the current population; NaN if none.
    real_feasible_ratio: list     # Current-population feasible fraction (vio <= FEASIBLE_TOL).
    archive_dec: np.ndarray       # Consumed-HF decision archive, shape (n_archive, dim_d).
    archive_real_obj: np.ndarray  # True objective for each archived decision.
    archive_real_vio: np.ndarray  # Raw true constraint violation for each archived decision.
    final_real_obj: np.ndarray    # True objective for each final-population member.
    final_real_vio: np.ndarray    # Raw true violation for each final-population member.
    pde_residual_final: float     # Final full PDE residual MSE (PINN/PINO/PIGP; NaN otherwise).
    pde_residual_linear_final: float  # Final linear-operator residual MSE (operator PIGP only; NaN otherwise).


class Experiment:
    def __init__(self, problem, method, protocol, data_path: str, seed: int):
        self.problem = problem
        self.method = method
        self.protocol = protocol
        self.data_path = data_path
        self.seed = seed

    def run(self, *, evaluate_final_residual=True) -> ExperimentResult:
        problem, method, protocol = self.problem, self.method, self.protocol

        # Initialize reference data, oracle, and evaluation grid in reproducible order.
        # Attach data-backed fields such as the F07 source term.
        problem.attach_data(self.data_path)
        reference = problem.load_reference(self.data_path)
        # Explicit parameter axes count unique complete parameter vectors.
        # None uses axis 2 for queries with at least three dimensions.
        oracle = HighFidelityOracle(reference,
                                    parameter_axes=problem.hf_parameter_axes)
        metric_grid = MetricGrid(problem, reference,
                                 protocol.mse_grid_nx, protocol.mse_grid_nt,
                                 getattr(protocol, "mse_grid_nmu", None))

        # Initialize the surrogate without fitting.
        method.setup(problem)
        # Self-managed SAEA loops require decision-level evaluation.
        if getattr(method, "needs_decision_evaluator", False):
            from evaluation.decision_eval import DecisionEvaluator
            method.bind_evaluator(DecisionEvaluator(problem, oracle, reference))

        # Float32 pool storage preserves reference-run numerics.
        dim_d = problem.decision_bounds.shape[0]
        dim_q = problem.query_bounds.shape[0]
        pool_dec = np.empty((0, dim_d), dtype=np.float32)
        pool_Q = np.empty((0, dim_q), dtype=np.float32)
        pool_u = np.empty((0,), dtype=np.float32)
        pool_f = np.empty((0,), dtype=np.float32)
        pool_vio = np.empty((0,), dtype=np.float32)
        # Rank official results using decisions whose HF evaluations were consumed.
        # Keep the archive in float64 so training-pool rounding cannot reduce best-F precision.
        archive_dec = np.empty((0, dim_d), dtype=np.float64)
        archive_real_obj = np.empty((0,), dtype=np.float64)
        archive_real_vio = np.empty((0,), dtype=np.float64)

        db_history = []
        train_calls = 0
        train_time = 0.0   # Cumulative surrogate training time.
        # Snapshot diagnostics: best true objective among current members with
        # vio <= FEASIBLE_TOL, plus the feasible fraction.
        real_watch_gen = []
        real_obj_best = []
        real_feasible_ratio = []

        # Surrogate objective and constraint callbacks in reduced form.
        # Protocol numerics override the problem query dtype (float32 for F09-F11).
        _qd = protocol.numerics.get("surrogate_query_dtype", "problem")
        if _qd == "problem":
            query_dtype = problem.surrogate_query_dtype
        elif _qd == "float32":
            query_dtype = np.float32
        else:  # float64 avoids rounding optimizer queries.
            query_dtype = None
        surr_provider = _SurrogateProvider(method.surrogate, query_dtype)

        def _batch_call(problem_fn, X, provider):
            """Collect queries, predict once, cache values, then evaluate each decision.

            Batch prediction is deterministic. Handle query-free callbacks explicitly;
            propagate other failures instead of silently changing execution paths."""
            X = np.atleast_2d(np.asarray(X, dtype=float))
            try:
                provider.begin_batch()
                first_pass = [problem_fn(x, provider) for x in X]
                has_queries = provider.flush_batch()
                if not has_queries:
                    provider.clear_batch()
                    return np.asarray(first_pass, dtype=float)
                return np.asarray([problem_fn(x, provider)
                                   for x in X], dtype=float)
            except Exception as exc:
                provider.clear_batch()
                raise RuntimeError("Batch problem evaluation failed; pointwise fallback was not used") from exc

        def fitness_fn(X):
            return _batch_call(problem.evaluate_fitness, X, surr_provider)

        def constraint_fn(X):
            return _batch_call(problem.violation, X, surr_provider)

        # The method update policy selects HFRequest points for the online hook.
        def online_hook(de, gen):
            nonlocal pool_dec, pool_Q, pool_u, pool_f, pool_vio
            nonlocal archive_dec, archive_real_obj, archive_real_vio
            nonlocal train_calls, train_time

            req = method.update_policy.request_hf_queries(
                de.pop, de.fitness, problem, gen
            )
            if req is None:
                return

            u_hf = oracle.evaluate(req.queries)                 # float64, (K,)

            # Log actual HF requests at each batch boundary.
            _param_kind = getattr(protocol, "budget_kind", "state") == "parameter"
            _block_mode = getattr(problem, "pool_f_block_mode", False)
            _n_rows = int(req.queries.shape[0])
            if _param_kind and not _block_mode:
                # For F09-F11, decisions and queries share (x,t,mu); group by mu instance.
                _n_dec = len({tuple(np.round(d[2:], 12)) for d in req.decisions})
            else:
                _n_dec = len({tuple(np.round(d, 12)) for d in req.decisions})
            _fe_now = (oracle.n_parameter_queries if _param_kind
                       else oracle.n_state_queries)
            _extra = f" ({_n_rows} state points)" if _n_rows != _n_dec else ""
            _hdr = "Initial HF queries" if int(gen) == 0 else "HF queries"
            print(f"[HF] gen {int(gen):3d} | {_hdr} {_n_dec} decisions{_extra} | "
                  f"FE {_fe_now}/{protocol.hf_budget}")

            # Protocol numerics override the dataset label precision.
            # Float32 datasets round labels before computing f/vio.
            _ld = protocol.numerics.get("label_dtype", "dataset")
            label_dtype = (reference.label_dtype if _ld == "dataset"
                           else np.dtype(_ld))
            u_lab = u_hf.astype(label_dtype)

            # Discard failed-solve labels before extending the archive.
            # The solve attempt already consumed one FE even though no label was produced.
            finite = np.isfinite(u_lab)
            if not finite.all():
                n_bad = int((~finite).sum())
                print(f"[WARN] Discarding {n_bad} nonfinite labels from failed solves in this batch")
                req_q = req.queries[finite]
                req_d = req.decisions[finite]
                u_lab = u_lab[finite]
            else:
                req_q = req.queries
                req_d = req.decisions
            n = req_q.shape[0]
            if n == 0:
                return   # All solves failed: consume the budget without fitting or taking a snapshot.

            # Compute f/vio in float64 using the problem-specific objective and constraint paths.
            # HFRequest provides decisions: population rows or inverse-mapped synthetic queries.
            f_sel = np.empty((n,), dtype=float)
            vio_sel = np.empty((n,), dtype=float)
            if getattr(problem, "pool_f_block_mode", False):
                # Blockwise objectives reconstruct J(theta) from complete query blocks.
                # Cache one objective per theta without additional solves.
                block_prov = _BlockProvider(req_q, u_lab, reference)
                j_cache = {}
                for i in range(n):
                    dkey = tuple(np.round(req_d[i], 12))
                    if dkey not in j_cache:
                        j_cache[dkey] = (
                            float(problem.evaluate_fitness_for_pool(
                                req_d[i], block_prov)),
                            float(problem.violation(req_d[i], block_prov)),
                        )
                    f_sel[i], vio_sel[i] = j_cache[dkey]
            else:
                for i in range(n):
                    d_i = req_d[i]
                    prov_i = _PointProvider(req_q[i], u_lab[i], reference)
                    f_sel[i] = float(problem.evaluate_fitness_for_pool(d_i, prov_i))
                    vio_sel[i] = float(problem.violation(d_i, prov_i))

            # Archive one row per true decision evaluation. Pointwise requests map one-to-one;
            # Blockwise requests expand one decision into multiple field queries.
            if _block_mode:
                archive_rows, seen_decisions = [], set()
                for i, decision in enumerate(req_d):
                    key = tuple(np.round(decision, 12))
                    if key not in seen_decisions:
                        seen_decisions.add(key)
                        archive_rows.append(i)
                archive_rows = np.asarray(archive_rows, dtype=int)
            else:
                archive_rows = np.arange(n, dtype=int)
            archive_dec = np.vstack([
                archive_dec, np.asarray(req_d[archive_rows], dtype=np.float64)
            ])
            archive_real_obj = np.concatenate([
                archive_real_obj, np.asarray(f_sel[archive_rows], dtype=np.float64)
            ])
            archive_real_vio = np.concatenate([
                archive_real_vio, np.asarray(vio_sel[archive_rows], dtype=np.float64)
            ])

            # Reference logging matches final_real_obj diagnostics, without FE or RNG use.
            # For blockwise objectives, use the reconstructed true J from j_cache because
            # direct reference evaluation would launch another solve.
            if _block_mode:
                for _dk, (_f, _v) in j_cache.items():
                    _zs = ", ".join(f"{_x:.4g}" for _x in _dk)
                    print(f"[HF]   z=[{_zs}]  f={_f:.6g}  "
                          f"vio={0.0 if _v == 0 else _v:.2e}")
            elif _param_kind:
                # For F09-F11, log one mu instance with its objective range and worst violation.
                # Pointwise values remain available in history.npz f/vio_data.
                _ref_prov = _ReferenceProvider(reference)
                _groups = {}
                for i in range(n):
                    _mk = tuple(np.round(req_d[i][2:], 12))
                    _f = float(problem.evaluate_fitness(req_d[i], _ref_prov))
                    _v = float(problem.violation(req_d[i], _ref_prov))
                    _g = _groups.setdefault(_mk, [[], 0.0])
                    _g[0].append(_f)
                    _g[1] = max(_g[1], _v)
                for _mk, (_fs, _vmax) in _groups.items():
                    _ms = ", ".join(f"{_x:.4g}" for _x in _mk)
                    print(f"[HF]   mu=[{_ms}]  {len(_fs)} state points | "
                          f"f∈[{min(_fs):.6g}, {max(_fs):.6g}] | "
                          f"vio_max={0.0 if _vmax == 0 else _vmax:.2e}")
            else:
                _ref_prov = _ReferenceProvider(reference)
                _log_items, _seen_dec = [], set()
                for i in range(n):
                    _dk = tuple(np.round(req_d[i], 12))
                    if _dk in _seen_dec:
                        continue
                    _seen_dec.add(_dk)
                    _log_items.append(
                        (_dk,
                         float(problem.evaluate_fitness(req_d[i], _ref_prov)),
                         float(problem.violation(req_d[i], _ref_prov))))
                for _dk, _f, _v in _log_items:
                    _zs = ", ".join(f"{_x:.4g}" for _x in _dk)
                    print(f"[HF]   z=[{_zs}]  f={_f:.6g}  "
                          f"vio={0.0 if _v == 0 else _v:.2e}")

            # Float32 pool storage preserves reference-run label rounding.
            pool_dec = np.vstack([pool_dec, req_d.astype(np.float32)])
            pool_Q = np.vstack([pool_Q, req_q.astype(np.float32)])
            pool_u = np.concatenate([pool_u, u_lab.astype(np.float32)])
            pool_f = np.concatenate([pool_f, f_sel.astype(np.float32)])
            pool_vio = np.concatenate([pool_vio, vio_sel.astype(np.float32)])

            # Fit, monitor MSE, then snapshot in reproducible order.
            import time as _time
            _t0 = _time.perf_counter()
            method.surrogate.fit(pool_Q, pool_u,
                                 gen=gen, maxgen=protocol.maxgen)
            train_calls += 1
            train_time += _time.perf_counter() - _t0   # Training runtime.
            mse200 = metric_grid.mse(method.surrogate)
            # Optional engineering-observable MSE; omitted when undefined.
            vmse = problem.metric_voltage_mse(method.surrogate, reference)

            # Snapshot keys x_data/t_data identify the first two query axes.
            # Append further axes using coordinate_names, e.g. mu_data for F09.
            snap = {
                "gen": int(gen),
                "eval_count": int(pool_Q.shape[0]),
                "mse200": float(mse200),
                "dec_data": pool_dec.copy(),
                "x_data": pool_Q[:, 0].copy(),
                "t_data": pool_Q[:, 1].copy(),
                "u_data": pool_u.copy(),
                "f_data": pool_f.copy(),
                "vio_data": pool_vio.copy(),
            }
            if vmse is not None:
                snap["voltage_mse"] = float(vmse)   # Optional observable metric; omitted when undefined.
            # PIGP numerical diagnostics: jitter retries and estimated reciprocal condition
            # number of K, especially for high-order F08 operators; not ranking metrics.
            diag = getattr(method.surrogate, "last_fit_diagnostics", None)
            if diag is not None:
                snap["gp_jitter"] = int(diag["jitter"])
                snap["gp_rcond"] = float(diag["rcond"])
            # Optional PINN diagnostics separate dimensionless data, IC, BC, and PDE losses.
            # Omit these optional fields for surrogates that do not expose them.
            pinn_diag = getattr(method.surrogate, "last_loss_components", None)
            if pinn_diag is not None:
                for key, value in pinn_diag.items():
                    snap[f"pinn_loss_{key}"] = float(value)
            for j, cname in enumerate(problem.physics.coordinate_names[2:], start=2):
                snap[f"{cname}_data"] = pool_Q[:, j].copy()
            db_history.append(snap)

            # Reference diagnostics consume neither FE nor RNG draws.
            # Some backends disable snapshot reference evaluation to avoid extra solves
            # outside the algorithm's allocated budget; see protocol budget_kind.
            if problem.real_watch_enabled:
                ref_prov = _ReferenceProvider(reference)
                objs = np.array([problem.evaluate_fitness(d, ref_prov) for d in de.pop])
                vios = np.array([problem.violation(d, ref_prov) for d in de.pop])
                feas = vios <= FEASIBLE_TOL
                real_watch_gen.append(int(gen))
                real_feasible_ratio.append(float(feas.mean()))
                real_obj_best.append(float(objs[feas].min()) if feas.any()
                                     else float("nan"))
                print(f"[REAL] gen {int(gen):3d} | population best true objective "
                      f"f={real_obj_best[-1]:.6g} | "
                      f"feasible fraction {real_feasible_ratio[-1]:.3f}")

        # The method's constraint policy supplies optimizer penalty parameters.
        optimizer = method.build_optimizer(
            problem, protocol, fitness_fn, constraint_fn, online_hook
        )
        pop, fit = optimizer.run()

        # Collect self-managed SAEA round histories and timing without online_hook.
        if getattr(method, "needs_decision_evaluator", False):
            train_calls += int(getattr(optimizer, "surrogate_train_calls", 0))
            train_time += float(getattr(optimizer, "surrogate_train_time", 0.0))
            surr_provider.total_time += float(
                getattr(optimizer, "surrogate_infer_time", 0.0)
            )

        # Self-managed drivers store true evaluations in their archives and populations;
        # their round histories already provide true-objective diagnostics.
        driver_hist = getattr(optimizer, "history", None)
        if driver_hist:
            db_history = driver_hist
            real_watch_gen = [h["gen"] for h in driver_hist]
            real_obj_best = [h["real_best"] for h in driver_hist]
            real_feasible_ratio = [h["feasible_ratio"] for h in driver_hist]

        # Final-population evaluation is diagnostic; official best objectives use the HF archive.
        ref_prov = _ReferenceProvider(reference)
        final_real_obj = np.array(
            [problem.evaluate_fitness(d, ref_prov) for d in pop])
        final_real_vio = np.array(
            [problem.violation(d, ref_prov) for d in pop])

        # Self-managed SAEA drivers expose their true-evaluation archives.
        # Compatibility with older drivers falls back to their returned true final sets.
        if getattr(method, "needs_decision_evaluator", False):
            archive_dec = np.asarray(
                getattr(optimizer, "archive_dec", pop), dtype=float
            )
            archive_real_obj = np.asarray(
                getattr(optimizer, "archive_obj", final_real_obj), dtype=float
            ).reshape(-1)
            archive_real_vio = np.asarray(
                getattr(optimizer, "archive_vio", final_real_vio), dtype=float
            ).reshape(-1)

        archive_dec = np.atleast_2d(np.asarray(archive_dec, dtype=float))
        archive_real_obj = np.asarray(archive_real_obj, dtype=float).reshape(-1)
        archive_real_vio = np.asarray(archive_real_vio, dtype=float).reshape(-1)
        if (archive_dec.shape[0] != archive_real_obj.size
                or archive_real_obj.size != archive_real_vio.size):
            raise RuntimeError(
                "HF archive decision/objective/violation row counts differ"
            )

        # Full PDE residuals: PINN autograd, PINO grid derivatives, PIGP closed-form derivatives.
        # Other surrogates report NaN. Operator PIGP also reports the linear residual,
        # corresponding to the covariance-conditioned operator block,
        # alongside the full nonlinear PDE residual.
        pde_residual_final = float("nan")
        pde_residual_linear_final = float("nan")
        if evaluate_final_residual and hasattr(method.surrogate, "_pde_residual"):
            from evaluation.metrics.surrogate import pde_residual_mse
            pde_residual_final = pde_residual_mse(method.surrogate, metric_grid.Q)
        elif evaluate_final_residual and hasattr(method.surrogate, "predict_with_derivatives"):
            from evaluation.metrics.surrogate import pigp_operator_residuals
            pde_residual_final, pde_residual_linear_final = \
                pigp_operator_residuals(method.surrogate, problem,
                                        metric_grid.Q)

        # State budgets require exact point counts for the paper protocol.
        # Parameter budgets limit request slots; the number
        # of distinct parameters may be lower after convergence, so only overspending fails.
        if getattr(protocol, "budget_kind", "state") == "parameter":
            if oracle.n_parameter_queries > protocol.hf_budget:
                print(f"[WARN] Distinct parameter count {oracle.n_parameter_queries} "
                      f"exceeds protocol budget {protocol.hf_budget}")
        elif oracle.n_state_queries != protocol.hf_budget:
            print(f"[WARN] HF count {oracle.n_state_queries} != protocol budget {protocol.hf_budget}"
                  f" (state budget)")

        return ExperimentResult(
            pop=pop,
            fitness=fit,
            db_history=db_history,
            n_hf_queries=oracle.n_hf_queries,
            n_state_queries=oracle.n_state_queries,
            n_parameter_queries=oracle.n_parameter_queries,
            hf_wall_time=oracle.wall_time,
            surrogate_train_calls=train_calls,
            surrogate_train_time=train_time,
            surrogate_infer_time=surr_provider.total_time,
            real_watch_gen=real_watch_gen,
            real_obj_best=real_obj_best,
            real_feasible_ratio=real_feasible_ratio,
            archive_dec=archive_dec,
            archive_real_obj=archive_real_obj,
            archive_real_vio=archive_real_vio,
            final_real_obj=final_real_obj,
            final_real_vio=final_real_vio,
            pde_residual_final=pde_residual_final,
            pde_residual_linear_final=pde_residual_linear_final,
        )
