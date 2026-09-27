# surrogates/rbfn_field.py
"""Data-only Gaussian radial-basis network for the PDE state field.

Normalize query coordinates to [0,1]^d without a denominator epsilon. Training
points become deterministic centers. Basis functions are phi(r)=exp(-r^2),
where r is normalized distance divided by the center width sigma_j. Widths use
np.partition at index k (sigma_k defaults to 2), including the zero self-distance,
and are bounded below by sigma_floor. Ridge output weights satisfy
w=(Phi.T@Phi+lambda*I)^(-1)@Phi.T@y, with default lambda=1e-6.

Remove exact duplicates, merge nearby centers, and skip fitting with fewer
than two points. Optional max_train_points uses deterministic uniform
subsampling with seed 20260821. Calculations use float64; no physics is read.
"""
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist

from surrogates.base import Surrogate


class RBFNSurrogate(Surrogate):
    name = "RBFN"

    def __init__(self):
        self.qmin = self.qmax = None
        self.sigma_k = 2          # Width uses the k-th nearest other center (zero self-distance occupies index 0).
        self.sigma_floor = 1e-4   # Minimum width guards against degenerate basis columns.
        self.reg = 1e-6           # Protocol-configurable ridge regularization.
        self.max_train_points = None
        self.centers = None       # Centers in normalized coordinates (training points).
        self.sigma = None         # Per-center widths sigma_j in normalized coordinates.
        self.w = None             # Output weights.
        self._dim = 0

    def setup(self, problem, seed: int, config: dict = None):
        config = config or {}
        qb = np.asarray(problem.query_bounds, dtype=float)
        self.qmin, self.qmax = qb[:, 0], qb[:, 1]
        self._dim = qb.shape[0]
        self.sigma_k = int(config.get("sigma_k", 2))
        self.sigma_floor = float(config.get("sigma_floor", 1e-4))
        self.reg = float(config.get("reg", 1e-6))
        # Optional deterministic training cap avoids cubic matrix cost for large training pools.
        self.max_train_points = config.get("max_train_points", None)
        self.seed = seed  # Accepted for the interface; fitting itself is deterministic.

    def _normalize(self, Q: np.ndarray) -> np.ndarray:
        Q = np.asarray(Q, dtype=float)
        return (Q - self.qmin) / (self.qmax - self.qmin)

    def fit(self, Q: np.ndarray, u: np.ndarray, **ctx):
        X = self._normalize(Q)
        y = np.asarray(u, dtype=float).reshape(-1)
        # Elitism/top-K selection can add the same query across generations.
        # Remove exact duplicates to avoid identical basis columns.
        X, idx = np.unique(X, axis=0, return_index=True)
        y = y[idx]   # Use lexicographic unique-row indices; sorting idx would misalign X and y.
        # Float32 pool queries can be separated by only ~1e-5 after normalization.
        # Tiny kNN widths for near-coincident centers can make basis columns nearly dependent
        # and the ridge system ill-conditioned (observed condition ~1e7, solve error O(1)).
        # Treat point pairs closer than sigma_floor as redundant centers.
        # Merge connected components by union-find; average coordinates and labels deterministically.
        tol = max(float(self.sigma_floor), 1e-9)
        pairs = cKDTree(X).query_pairs(tol)
        if pairs:
            parent = np.arange(X.shape[0])
            def _find(i):
                while parent[i] != i:
                    parent[i] = parent[parent[i]]
                    i = parent[i]
                return i
            for i, j in pairs:
                ri, rj = _find(i), _find(j)
                if ri != rj:
                    parent[max(ri, rj)] = min(ri, rj)
            labels = np.array([_find(i) for i in range(X.shape[0])])
            Xm, ym = [], []
            for lab in np.unique(labels):
                m = labels == lab
                Xm.append(X[m].mean(axis=0))
                ym.append(y[m].mean())
            X, y = np.asarray(Xm), np.asarray(ym)
        if self.max_train_points is not None and X.shape[0] > int(self.max_train_points):
            # Fixed-seed uniform subsampling matches the other field surrogates.
            rng = np.random.default_rng(20260821)
            idx2 = rng.choice(X.shape[0], int(self.max_train_points), replace=False)
            X, y = X[idx2], y[idx2]
        # Fewer than two points cannot determine distance-based widths; skip fitting.
        if X.shape[0] < 2:
            self.centers = None
            self.w = None
            return
        n = X.shape[0]
        self.centers = X
        # Select distance at zero-based index k, with self-distance at index 0.
        # Thus k=2 selects the second other center when enough centers are available.
        k = min(self.sigma_k, n - 1)
        d = cdist(X, X)
        self.sigma = np.partition(d, k, axis=0)[k]
        self.sigma = np.maximum(self.sigma, self.sigma_floor)
        # Phi[i,j]=exp(-(d_ij/sigma_j)^2); per-center widths make Phi asymmetric.
        # Ridge output layer: w=(Phi.T@Phi+lambda*I)^(-1)@Phi.T@y.
        Phi = np.exp(-(d / self.sigma) ** 2)
        A = Phi.T @ Phi + self.reg * np.eye(n)
        self.w = np.linalg.solve(A, Phi.T @ y)

    def predict(self, Q: np.ndarray) -> np.ndarray:
        X = self._normalize(np.asarray(Q, dtype=float).reshape(-1, self.qmin.size))
        Phi = np.exp(-(cdist(X, self.centers) / self.sigma) ** 2)
        return Phi @ self.w
