# methods/policies.py
"""HF sampling policies used by optimization methods.

Methods/policies choose query locations and counts. Protocols set allowed
budgets, and the oracle counts actual consumption. A policy returns an
HFRequest containing query points, independent of how they were selected.
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class HFRequest:
    """HF labeling request for one online update."""
    queries: np.ndarray      # (K, dim_query) query points.
    owner_idx: np.ndarray    # (K,) source population index per query; -1 denotes a synthetic point.
                             # Synthetic top-up points are not population members; indices record provenance.
    decisions: np.ndarray    # (K, dim_decision) complete decision vector for each query.
                             # Population row, or problem.query_to_decision for synthetic points.


class PeriodicTopK:
    """Periodic top-K online updates (paper Algorithm 1).

    At gen=0, sample init_points from the initial population with a fresh
    default_rng(seed). At positive multiples of interval, select the k best
    penalized-fitness individuals; otherwise return None (no HF request).
    Oversample by candidate_factor, filter with problem.is_valid_query, and
    retain K valid points. If needed, replenish using
    problem.sample_query_candidates and a fresh default_rng(seed+gen+1000).
    candidate_factor=1 preserves the baseline RNG behavior.
    """

    def __init__(self, interval: int, k: int, init_points: int, seed: int,
                 candidate_factor: int = 1):
        self.interval = interval
        self.k = k
        self.init_points = init_points
        self.seed = seed
        self.candidate_factor = int(candidate_factor)

    def request_hf_queries(self, population: np.ndarray, fitness: np.ndarray,
                           problem, gen: int):
        pop = population
        if gen == 0:
            # At initialization, init_points<=0 selects the whole population.
            K = int(self.init_points)
            if K <= 0:
                K = int(pop.shape[0])
            K = min(K, int(pop.shape[0]))
            rng = np.random.default_rng(self.seed)
            cand_num = min(max(self.candidate_factor * K, K), pop.shape[0])
            idx = rng.choice(pop.shape[0], cand_num, replace=False)
        elif gen % self.interval == 0:
            K = int(self.k)
            cand_num = min(max(self.candidate_factor * K, K), pop.shape[0])
            idx = np.argsort(fitness)[:cand_num]
        else:
            return None

        cand_dec = np.asarray(pop[idx], dtype=float)
        cand_Q = np.asarray(
            [problem.decision_to_query(d) for d in cand_dec], dtype=float
        )
        ok = np.asarray(problem.is_valid_query(cand_Q), dtype=bool)

        sel_dec = cand_dec[ok][:K]
        sel_Q = cand_Q[ok][:K]
        sel_owner = np.asarray(idx, dtype=int)[ok][:K]

        need = K - sel_Q.shape[0]
        if need > 0:
            # Top up to K valid candidates when domain filtering leaves a shortfall.
            # Create this RNG only as needed; do not consume the existing stream.
            rng2 = np.random.default_rng(self.seed + int(gen) + 1000)
            extra_Q = []
            while need > 0:
                m = max(need * 4, 1024)
                cand2 = problem.sample_query_candidates(m, rng2)
                ok2 = np.asarray(problem.is_valid_query(cand2), dtype=bool)
                take = min(need, int(ok2.sum()))
                if take > 0:
                    extra_Q.append(cand2[ok2][:take])
                    need -= take
            extra_Q = np.concatenate(extra_Q, axis=0)
            extra_dec = np.asarray(problem.query_to_decision(extra_Q), dtype=float)
            sel_Q = np.concatenate([sel_Q, extra_Q], axis=0)
            sel_dec = np.concatenate([sel_dec, extra_dec], axis=0)
            sel_owner = np.concatenate(
                [sel_owner, -np.ones(extra_Q.shape[0], dtype=int)]
            )

        return HFRequest(queries=sel_Q, owner_idx=sel_owner, decisions=sel_dec)


class TopKMuRandomXT:
    """Online sampling for parameterized decisions (x,t,mu).

    At gen=0, joint_random draws init_points uniformly without the population.
    At positive multiples of interval, take mu from the k best penalized-fitness
    individuals and pair each with xt_per_mu random (x,t), totaling k*xt_per_mu
    queries. A persistent default_rng(seed) is consumed across calls.
    All queries are synthetic (owner_idx=-1), normalized for periodic mapping
    or clipping, and converted to decisions by query_to_decision.

    init_mode="joint_random" uses the paper sampling scheme.
    init_mode="hierarchical" draws init_mu parameters, each paired with
    xt_per_mu state points: N_mu PDE instances times N_xt labels per instance.
    For each parameter, the stream draws mu, then the x vector, then t.
    """

    def __init__(self, interval: int, k: int, init_points: int, seed: int,
                 xt_per_mu: int, init_mode: str = "joint_random"):
        self.interval = interval
        self.k = k
        self.init_points = init_points
        self.seed = seed
        self.xt_per_mu = xt_per_mu
        self.init_mode = init_mode
        self.rng = np.random.default_rng(seed)   # Persistent RNG stream; do not recreate per call.

    def request_hf_queries(self, population: np.ndarray, fitness: np.ndarray,
                           problem, gen: int):
        qb = np.asarray(problem.query_bounds, dtype=float)
        if gen == 0:
            if self.init_mode == "hierarchical":
                # Hierarchical mode interprets init_points as the initial PDE instance count N_mu.
                N_mu = int(max(1, self.init_points))
                M = int(self.xt_per_mu)
                mus_list, xs_list, ts_list = [], [], []
                for _ in range(N_mu):
                    mu_i = qb[2, 0] + (qb[2, 1] - qb[2, 0]) * self.rng.random()
                    mus_list.append(np.full(M, mu_i))
                    xs_list.append(qb[0, 0] + (qb[0, 1] - qb[0, 0]) * self.rng.random(M))
                    ts_list.append(qb[1, 0] + (qb[1, 1] - qb[1, 0]) * self.rng.random(M))
                mus = np.concatenate(mus_list)
                xs = np.concatenate(xs_list)
                ts = np.concatenate(ts_list)
            else:
                N0 = int(max(1, self.init_points))
                # Draw with rng.random per axis; switching to uniform changes baseline RNG consumption.
                xs = qb[0, 0] + (qb[0, 1] - qb[0, 0]) * self.rng.random(N0)
                ts = qb[1, 0] + (qb[1, 1] - qb[1, 0]) * self.rng.random(N0)
                mus = qb[2, 0] + (qb[2, 1] - qb[2, 0]) * self.rng.random(N0)
        elif gen % self.interval == 0:
            K = int(min(self.k, population.shape[0]))
            M = int(self.xt_per_mu)
            idx = np.argsort(fitness)[:K]
            mu_sel = population[idx, 2].astype(float)
            N = K * M
            mus = np.repeat(mu_sel, M)
            xs = qb[0, 0] + (qb[0, 1] - qb[0, 0]) * self.rng.random(N)
            ts = qb[1, 0] + (qb[1, 1] - qb[1, 0]) * self.rng.random(N)
        else:
            return None

        Q = np.column_stack([xs, ts, mus])
        Q = np.asarray(problem.normalize_queries(Q), dtype=float)
        dec = np.asarray(problem.query_to_decision(Q), dtype=float)
        return HFRequest(
            queries=Q,
            owner_idx=-np.ones(Q.shape[0], dtype=int),
            decisions=dec,
        )


class TopKFullDecision:
    """Periodically select the highest-ranked complete decisions.

    Retain each selected (x,t,mu) rather than pairing its mu with random state
    points. Initialization samples init_points complete population decisions
    without replacement using a fixed seed, as in PeriodicTopK.
    normalize_queries maps periodic endpoints/data boundaries to valid reference
    coordinates. Recover decisions from those normalized queries so labels and
    decisions refer to the same physical points. Each request contains exactly
    K complete decisions and no additional random field points.
    """

    def __init__(self, interval: int, k: int, init_points: int, seed: int):
        self.interval = int(interval)
        self.k = int(k)
        self.init_points = int(init_points)
        self.seed = int(seed)

    def request_hf_queries(self, population: np.ndarray, fitness: np.ndarray,
                           problem, gen: int):
        pop = np.asarray(population, dtype=float)
        if gen == 0:
            K = min(max(1, self.init_points), int(pop.shape[0]))
            rng = np.random.default_rng(self.seed)
            idx = rng.choice(pop.shape[0], K, replace=False)
        elif gen % self.interval == 0:
            K = min(self.k, int(pop.shape[0]))
            idx = np.argsort(np.asarray(fitness, dtype=float), kind="stable")[:K]
        else:
            return None

        selected = pop[np.asarray(idx, dtype=int)]
        Q = np.asarray(
            [problem.decision_to_query(d) for d in selected], dtype=float
        )
        Q = np.asarray(problem.normalize_queries(Q), dtype=float)
        valid = np.asarray(problem.is_valid_query(Q), dtype=bool)
        if valid.shape != (K,) or not valid.all():
            raise RuntimeError(
                "TopKFullDecision selected decisions contain invalid queries after normalization"
            )
        decisions = np.asarray(problem.query_to_decision(Q), dtype=float)
        return HFRequest(
            queries=Q,
            owner_idx=np.asarray(idx, dtype=int),
            decisions=decisions,
        )


class TopKThetaBlock:
    """Blockwise sampling: top-K parameter values with complete query blocks.

    Reconstructing J(theta) requires the full surface grid for that theta.
    problem.policy_query_block supplies every surface point plus random interior
    points so pool objectives need no additional solves. At gen=0, draw
    init_points random theta vectors using per-axis uniform draws. On update
    generations, select the k best penalized-fitness decisions. Keep the RNG
    stream persistent across calls.
    """

    def __init__(self, interval: int, k: int, init_points: int, seed: int,
                 interior_points: int):
        self.interval = interval
        self.k = k
        self.init_points = init_points
        self.seed = seed
        self.interior_points = interior_points
        self.rng = np.random.default_rng(seed)   # Persistent RNG stream; do not recreate per call.

    def request_hf_queries(self, population: np.ndarray, fitness: np.ndarray,
                           problem, gen: int):
        if gen == 0:
            K = int(self.init_points)
            db = np.asarray(problem.decision_bounds, dtype=float)
            cols = [self.rng.uniform(db[j, 0], db[j, 1], size=K)
                    for j in range(db.shape[0])]
            thetas = np.column_stack(cols)
        elif gen % self.interval == 0:
            K = int(min(self.k, population.shape[0]))
            idx = np.argsort(fitness)[:K]
            thetas = np.asarray(population[idx], dtype=float)
        else:
            return None

        Q, dec = problem.policy_query_block(thetas, self.rng,
                                            self.interior_points)
        return HFRequest(
            queries=Q,
            owner_idx=-np.ones(Q.shape[0], dtype=int),
            decisions=dec,
        )


@dataclass
class DynamicPenalty:
    """Penalty parameters for a linear schedule from min to max.

    The optimizer implements the schedule; this object stores the method's
    selected parameters.
    """
    min: float
    max: float
