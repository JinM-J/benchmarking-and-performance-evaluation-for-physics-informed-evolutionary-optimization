# evaluation/decision_eval.py
"""Evaluate complete decisions against the reference state.

Pointwise problems charge one state query per decision. Integral problems
charge the anchored decision query and use uncharged reference quadrature,
matching _PointProvider. Blockwise extensions query a full block per decision.
Returns objectives, per-constraint components and aggregate violations."""
import numpy as np

from core.experiment import _PointProvider, _BlockProvider


class DecisionEvaluator:
    def __init__(self, problem, oracle, reference):
        self.problem = problem
        self.oracle = oracle
        self.reference = reference

    def evaluate(self, decisions: np.ndarray):
        """decisions (n, dim_d) -> (f (n,), cons (n, nc), vio (n,))。"""
        D = np.atleast_2d(np.asarray(decisions, dtype=float))
        n = D.shape[0]
        f = np.empty(n)
        cons = []
        vio = np.empty(n)

        block_mode = getattr(self.problem, "pool_f_block_mode", False)
        for i, d in enumerate(D):
            if block_mode:
                # Evaluate a query block; count one parameter vector per solve.
                Q = self.problem._surface_queries(d)
                u = self.oracle.evaluate(Q)
                prov = _BlockProvider(Q, u, self.reference)
            else:
                q = self.problem.decision_to_query(d)
                u = self.oracle.evaluate(np.atleast_2d(q))
                prov = _PointProvider(np.atleast_2d(q)[0], float(u[0]),
                                      self.reference)
            f[i] = float(self.problem.evaluate_fitness_for_pool(d, prov))
            comps = self.problem.constraint_components(d, prov)
            cons.append([c["value"] for c in comps])
            vio[i] = float(self.problem.violation(d, prov))

        nc = len(cons[0]) if cons else 0
        C = np.array(cons, dtype=float) if nc else np.empty((n, 0))
        return f, C, vio

    def evaluate_queries(self, decisions: np.ndarray, queries: np.ndarray,
                         label_dtype=None):
        """Evaluate aligned decision/query batches with one oracle call.

        Distinct parameter values are counted within the query batch. Each row uses
        _PointProvider for objective and constraint evaluation, as in the training pool.
        decisions has shape (n, dim_decision); queries has shape (n, dim_query).
        label_dtype follows protocol.numerics. Returns (f, components, violation, u),
        with u rounded to the label precision actually stored in the pool."""
        if getattr(self.problem, "pool_f_block_mode", False):
            raise ValueError(
                "evaluate_queries supports pointwise anchors only; use evaluate for block-state problems"
            )

        D = np.atleast_2d(np.asarray(decisions, dtype=float))
        Q = np.atleast_2d(np.asarray(queries, dtype=float))
        if D.shape[0] != Q.shape[0]:
            raise ValueError(
                f"Decision/query row counts differ: {D.shape[0]} != {Q.shape[0]}"
            )
        if D.shape[1] != np.asarray(self.problem.decision_bounds).shape[0]:
            raise ValueError("Decision column count does not match problem.decision_bounds")
        if Q.shape[1] != np.asarray(self.problem.query_bounds).shape[0]:
            raise ValueError("Query column count does not match problem.query_bounds")
        if not np.all(self.problem.is_valid_query(Q)):
            raise ValueError("True-evaluation request contains points rejected by problem.is_valid_query")

        u = np.asarray(self.oracle.evaluate(Q)).reshape(-1)
        if label_dtype is not None:
            u = u.astype(label_dtype)
        if not np.isfinite(u).all():
            raise RuntimeError("True evaluation returned nonfinite state labels")

        n = D.shape[0]
        f = np.empty(n, dtype=float)
        cons = []
        vio = np.empty(n, dtype=float)
        for i, d in enumerate(D):
            prov = _PointProvider(Q[i], float(u[i]), self.reference)
            f[i] = float(self.problem.evaluate_fitness_for_pool(d, prov))
            comps = self.problem.constraint_components(d, prov)
            cons.append([c["value"] for c in comps])
            vio[i] = float(self.problem.violation(d, prov))

        nc = len(cons[0]) if cons else 0
        C = np.asarray(cons, dtype=float) if nc else np.empty((n, 0))
        return f, C, vio, u
