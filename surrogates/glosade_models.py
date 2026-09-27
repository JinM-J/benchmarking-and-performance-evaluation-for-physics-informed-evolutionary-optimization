"""GRNN and cubic RBF models internal to GLoSADE.

Inputs are decisions, scalar objectives and explicit constraints, without PDE
states, residuals or Problem.physics access. Following Wang et al. (2019) and
the authors' PlatEMO implementation, global fitness search uses GRNN with
spread=0.5; local search and uncertainty use cubic RBFs with affine tails.
"""
from __future__ import annotations

import numpy as np


def _as_2d(X, *, name: str) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    if X.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional array")
    if not np.isfinite(X).all():
        raise ValueError(f"{name} contains nonfinite values")
    return X


def _pairwise_distance(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Stable row-wise Euclidean distances with shape (len(A), len(B))."""
    A = _as_2d(A, name="A")
    B = _as_2d(B, name="B")
    if A.shape[1] != B.shape[1]:
        raise ValueError("Distance-matrix inputs have different feature dimensions")
    d2 = (np.sum(A * A, axis=1)[:, None]
          + np.sum(B * B, axis=1)[None, :]
          - 2.0 * A @ B.T)
    return np.sqrt(np.maximum(d2, 0.0))


class GLoSADEGRNN:
    """Deterministic normalized Gaussian RBF equivalent to MATLAB newgrnn."""

    def __init__(self, spread: float = 0.5):
        self.spread = float(spread)
        if not np.isfinite(self.spread) or self.spread <= 0.0:
            raise ValueError("GRNN spread must be positive")
        self.X = None
        self.Y = None

    def fit(self, X, Y):
        X = _as_2d(X, name="GRNN X")
        Y = _as_2d(Y, name="GRNN Y")
        if len(X) != len(Y):
            raise ValueError("GRNN X/Y sample counts differ")
        if len(X) == 0:
            raise ValueError("GRNN requires at least one sample")
        self.X = X.copy()
        self.Y = Y.copy()
        return self

    def predict(self, Q) -> np.ndarray:
        if self.X is None:
            raise RuntimeError("GRNN has not been fitted")
        Q = _as_2d(Q, name="GRNN Q")
        if Q.shape[1] != self.X.shape[1]:
            raise ValueError("Invalid GRNN query dimension")

        # MATLAB newgrnn/newpnn radbas scales distances by 0.8326/spread.
        # 0.8326 approximates sqrt(log(2)); subtracting each row maximum log-weight
        # improves stability without changing the normalized weighted mean.
        dist2 = _pairwise_distance(Q, self.X) ** 2
        logits = -np.log(2.0) * dist2 / (self.spread ** 2)
        logits -= np.max(logits, axis=1, keepdims=True)
        weights = np.exp(logits)
        weights /= np.sum(weights, axis=1, keepdims=True)
        return weights @ self.Y


class CubicRBF:
    """Local GLoSADE cubic RBF interpolator with constant and linear polynomial tails."""

    def __init__(self):
        self.x_min = None
        self.x_max = None
        self.y_min = None
        self.y_max = None
        self.nodes = None
        self.alpha = None
        self.beta = None

    @staticmethod
    def _normalize(A, lo, hi):
        A = np.asarray(A, dtype=float).copy()
        varying = hi != lo
        if np.any(varying):
            A[:, varying] = (
                2.0 * (A[:, varying] - lo[varying])
                / (hi[varying] - lo[varying]) - 1.0
            )
        return A

    def fit(self, X, Y):
        X = _as_2d(X, name="cubic RBF X")
        Y = _as_2d(Y, name="cubic RBF Y")
        if len(X) != len(Y):
            raise ValueError("Cubic RBF X/Y sample counts differ")
        if len(X) == 0:
            raise ValueError("Cubic RBF requires at least one sample")

        self.x_min = np.min(X, axis=0)
        self.x_max = np.max(X, axis=0)
        self.y_min = np.min(Y, axis=0)
        self.y_max = np.max(Y, axis=0)
        Xn = self._normalize(X, self.x_min, self.x_max)
        Yn = self._normalize(Y, self.y_min, self.y_max)

        n, dim = X.shape
        phi = _pairwise_distance(Xn, Xn) ** 3
        poly = np.column_stack([np.ones(n), Xn])
        system = np.block([
            [phi, poly],
            [poly.T, np.zeros((dim + 1, dim + 1))],
        ])
        rhs = np.vstack([Yn, np.zeros((dim + 1, Y.shape[1]))])
        theta = np.linalg.pinv(system) @ rhs
        self.nodes = Xn
        self.alpha = theta[:n]
        self.beta = theta[n:]
        return self

    def predict(self, Q) -> np.ndarray:
        if self.nodes is None:
            raise RuntimeError("Cubic RBF has not been fitted")
        Q = _as_2d(Q, name="cubic RBF Q")
        if Q.shape[1] != self.nodes.shape[1]:
            raise ValueError("Invalid cubic RBF query dimension")
        Qn = self._normalize(Q, self.x_min, self.x_max)
        phi = _pairwise_distance(Qn, self.nodes) ** 3
        out = phi @ self.alpha + np.column_stack(
            [np.ones(len(Qn)), Qn]
        ) @ self.beta
        varying = self.y_max != self.y_min
        if np.any(varying):
            out[:, varying] = (
                (self.y_max[varying] - self.y_min[varying]) / 2.0
                * (out[:, varying] + 1.0) + self.y_min[varying]
            )
        if np.any(~varying):
            out[:, ~varying] = self.y_min[~varying]
        return out


def cubic_rbf_uncertainty(Q, archive_X, k: int) -> np.ndarray:
    """Candidate uncertainty from the GLoSADE KNN cubic-RBF power function.

    Each candidate selects K nearest neighbors and evaluates
    abs(-phi(q) @ pinv(Phi) @ phi(q).T). The small-budget adaptation caps K
    by archive size; no objective values or physics information enter this score.
    """
    Q = _as_2d(Q, name="uncertainty Q")
    archive_X = _as_2d(archive_X, name="uncertainty archive_X")
    if Q.shape[1] != archive_X.shape[1]:
        raise ValueError("Uncertainty queries and archive have different dimensions")
    k_eff = min(int(k), len(archive_X))
    if k_eff <= 0:
        raise ValueError("Uncertainty k must be positive")

    all_dist = _pairwise_distance(Q, archive_X)
    sigma2 = np.empty(len(Q), dtype=float)
    for i, q in enumerate(Q):
        near = np.argsort(all_dist[i], kind="stable")[:k_eff]
        Xn = archive_X[near]
        lo = np.min(Xn, axis=0)
        hi = np.max(Xn, axis=0)
        nodes = CubicRBF._normalize(Xn, lo, hi)
        qn = CubicRBF._normalize(q.reshape(1, -1), lo, hi)
        phi_matrix = _pairwise_distance(nodes, nodes) ** 3
        phi_q = _pairwise_distance(qn, nodes) ** 3
        try:
            solved = np.linalg.solve(phi_matrix, phi_q.T)
        except np.linalg.LinAlgError:
            # Duplicate archive points can make Phi singular; pinv is a deterministic safeguard.
            solved = np.linalg.pinv(phi_matrix) @ phi_q.T
        sigma2[i] = abs(float((-phi_q @ solved)[0, 0]))
    return sigma2
