# surrogates/rbfn.py
"""Ji et al. SAEA surrogate bundle: objective and per-constraint cubic RBFNs.

Models f(x), g_i(x) and h_j(x) on decision vectors rather than the PDE field
u(Q), so it does not inherit the field-level Surrogate interface. Uses SciPy
RBFInterpolator with phi(r)=r^3, degree=1 polynomial tails and deterministic
linear solves. Normalize decisions to [0,1]^d, remove duplicates and enforce
minimum sample counts to avoid singular interpolation matrices.
"""
import numpy as np
from scipy.interpolate import RBFInterpolator
from scipy.special import comb


class JiRBFNChannel:
    """Single cubic RBFN channel with fit(X,y) / predict(X)."""

    def __init__(self, qmin, qmax):
        self.qmin = np.asarray(qmin, dtype=float)
        self.qmax = np.asarray(qmax, dtype=float)
        self._scale = self.qmax - self.qmin
        if np.any(self._scale <= 0):
            raise ValueError("Ji RBFN requires positive bound spans in every decision dimension")
        self._dim = self.qmin.size
        self.model = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = (np.asarray(X, dtype=float) - self.qmin) / self._scale
        y = np.asarray(y, dtype=float).reshape(-1)
        # Remove duplicate decisions, retaining the first label; duplicates make interpolation singular.
        # idx follows lexicographic unique-row order; index y with idx, not np.sort(idx).
        X, idx = np.unique(X, axis=0, return_index=True)
        y = y[idx]
        n_req = int(comb(1 + self._dim, self._dim))   # Cubic default degree=1.
        if X.shape[0] < n_req:
            raise ValueError(
                f"Ji cubic RBFN in {self._dim} dimensions requires at least {n_req} unique decisions; "
                f"received {X.shape[0]}"
            )
        self.model = RBFInterpolator(X, y, kernel="cubic")

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Ji RBFN has not been fitted successfully")
        X = (np.asarray(X, dtype=float) - self.qmin) / self._scale
        return np.asarray(self.model(X), dtype=float).reshape(-1)


class JiSurrogateBundle:
    """One objective channel and nc individual constraint channels.

    Preserve problem.constraint_components order. Record kind/tol for each
    channel to compute feasibility and CV.
    """

    def __init__(self, bounds: np.ndarray, cons_meta: list):
        b = np.asarray(bounds, dtype=float)
        self.f_channel = JiRBFNChannel(b[:, 0], b[:, 1])
        self.cons_meta = list(cons_meta)          # [{"kind","tol"}, ...] in fixed channel order.
        self.cons_channels = [JiRBFNChannel(b[:, 0], b[:, 1])
                              for _ in self.cons_meta]

    def fit(self, X: np.ndarray, f: np.ndarray, C: np.ndarray):
        """Fit X (n,d), objective f (n,), and constraints C (n,nc); nc=0 is unconstrained."""
        self.f_channel.fit(X, f)
        for j, ch in enumerate(self.cons_channels):
            ch.fit(X, C[:, j])

    def predict_f(self, X: np.ndarray) -> np.ndarray:
        return self.f_channel.predict(X)

    def predict_cons(self, X: np.ndarray) -> np.ndarray:
        """Predict constraints with shape (n,nc); return (n,0) when nc=0."""
        if not self.cons_channels:
            return np.empty((np.asarray(X).shape[0], 0))
        return np.column_stack([ch.predict(X) for ch in self.cons_channels])

    # Derived quantities under the Ji constraint convention.
    def predict_constraint_violations(self, X: np.ndarray) -> np.ndarray:
        """Nonnegative per-constraint violations for Algorithm 2 channel-wise consensus."""
        C = self.predict_cons(X)
        if C.shape[1] == 0:
            return C
        V = np.empty_like(C)
        for j, meta in enumerate(self.cons_meta):
            if meta["kind"] == "g":
                V[:, j] = np.maximum(C[:, j] - meta["tol"], 0.0)
            else:
                V[:, j] = np.maximum(np.abs(C[:, j]) - meta["tol"], 0.0)
        return V

    def predict_cv(self, X: np.ndarray) -> np.ndarray:
        """Predict CV=sum(max(g,0))+sum(max(abs(h)-tol,0)) using declared tolerances."""
        V = self.predict_constraint_violations(X)
        if V.shape[1] == 0:
            return np.zeros(np.asarray(X).shape[0])
        return np.sum(V, axis=1)

    def predict_feasible(self, X: np.ndarray) -> np.ndarray:
        """Predict feasibility by checking every constraint channel against its tolerance."""
        C = self.predict_cons(X)
        n = np.asarray(X).shape[0]
        if C.shape[1] == 0:
            return np.ones(n, dtype=bool)
        ok = np.ones(n, dtype=bool)
        for j, meta in enumerate(self.cons_meta):
            if meta["kind"] == "g":
                ok &= C[:, j] <= meta["tol"]
            else:
                ok &= np.abs(C[:, j]) <= meta["tol"]
        return ok
