# surrogates/kernels/base.py
"""Anisotropic RBF kernel and closed-form mixed partial derivatives.

For k(x,x') = sigma2 * prod_a exp(-(x_a-x'_a)^2/(2*l_a^2)),
the operator-informed GP requires derivatives with per-axis orders p,q.
Using probabilists' Hermite polynomials, He_{n+1}=z*He_n-n*He_{n-1}:

    d_x^p d_x'^q g_a = (-1)^p * l_a^(-(p+q)) * He_{p+q}(Delta_a/l_a) * g_a
    g_a = exp(-Delta_a^2/(2*l_a^2)), Delta_a = x_a-x'_a.

The sign follows from d_x=d_Delta and d_x'=-d_Delta:
(-1)^q * (-1)^(p+q) = (-1)^p. Multiply per-axis derivatives and sigma2.
Applying the KS fourth derivative on both sides needs order eight on one axis.
These kernels support the custom joint block covariance and block noise model.
"""
import numpy as np


def hermite_he(n: int, z: np.ndarray) -> np.ndarray:
    """Probabilists' Hermite polynomial He_n(z); return 1 for n<=0."""
    z = np.asarray(z, dtype=float)
    if n <= 0:
        return np.ones_like(z)
    h0, h1 = np.ones_like(z), z.copy()
    if n == 1:
        return h1
    for k in range(1, n):
        h0, h1 = h1, z * h1 - k * h0
    return h1


def rbf(X: np.ndarray, Xp: np.ndarray, ls: np.ndarray, sigma2: float
        ) -> np.ndarray:
    """Anisotropic RBF: X (n,d), Xp (m,d) -> kernel matrix (n,m)."""
    D2 = (((X[:, None, :] - Xp[None, :, :]) / ls) ** 2).sum(-1)
    return sigma2 * np.exp(-0.5 * D2)


def rbf_mixed_deriv(X: np.ndarray, Xp: np.ndarray,
                    p: tuple, q: tuple,
                    ls: np.ndarray, sigma2: float) -> np.ndarray:
    """Mixed derivative d_x^p d_x'^q k(X,Xp); p/q give each axis order (length d)."""
    out = np.full((X.shape[0], Xp.shape[0]), sigma2)
    for a in range(X.shape[1]):
        pa, qa = p[a], q[a]
        delta = (X[:, None, a] - Xp[None, :, a]) / ls[a]      # Δ/l
        g = np.exp(-0.5 * delta ** 2)
        factor = ((-1.0) ** pa) * (ls[a] ** -(pa + qa)) \
            * hermite_he(pa + qa, delta) * g
        out = out * factor
    return out
