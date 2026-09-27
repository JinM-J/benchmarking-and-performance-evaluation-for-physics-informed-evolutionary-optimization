# surrogates/kernels/physics_kernel.py
"""Joint GP covariance under a linear differential operator.

For u~GP(0,k) and linear L, the covariance blocks of u and Lu are:
    K_uu = k(x,x')
    K_uL = L_{x'} k(x,x')
    K_Lu = L_x k(x,x') = K_uL^T
    K_LL = L_x L_{x'} k(x,x').

The problem supplies LinearOperator terms. PIGP converts derivatives to
normalized coordinates through d/dx_phys = span_a^(-1) d/dx_norm, while
coefficient functions are evaluated at physical coordinates.
For nonlinear PDEs, L is the exact linear part: e.g. Allen-Cahn retains
+5u and omits -5u^3. The operator-kernel identity requires a linear L.
"""
import numpy as np

from surrogates.kernels.base import rbf, rbf_mixed_deriv


class OperatorKernel:
    """Assemble covariance blocks from normalized linear-operator terms.

    terms_norm is a list of (deriv, coef_fn). deriv gives each query axis's
    derivative order (length dim). coef_fn maps X_norm (n,d) to (n,), including
    normalization factors after evaluating coefficients in physical coordinates.
    """

    def __init__(self, terms_norm):
        self.terms = terms_norm

    def k_uu(self, X, Xp, ls, sigma2):
        return rbf(X, Xp, ls, sigma2)

    def k_uL(self, X, Xp, ls, sigma2):
        """L_{x'} k(X,Xp), shape (n,m); k_duL with zero left derivative order."""
        dim = X.shape[1]
        return self.k_duL(X, Xp, tuple(0 for _ in range(dim)), ls, sigma2)

    def k_duL(self, X, Xp, p_left, ls, sigma2):
        """d_x^p L_{x'} k(X,Xp), shape (n,m).

        Posterior-mean derivative prediction requires both d_x^p k(x*,X_u)
        and d_x^p L_{x'} k(x*,X_L).
        """
        out = np.zeros((X.shape[0], Xp.shape[0]))
        for deriv, coef_fn in self.terms:
            c = np.asarray(coef_fn(Xp), dtype=float)          # (m,)
            out += rbf_mixed_deriv(X, Xp, p_left,
                                   deriv, ls, sigma2) * c[None, :]
        return out

    def k_LL(self, X, Xp, ls, sigma2):
        """L_x L_{x'} k(X, Xp)：(n, m)。"""
        out = np.zeros((X.shape[0], Xp.shape[0]))
        for ds, cs in self.terms:
            cx = np.asarray(cs(X), dtype=float)               # (n,)
            for dt, ct in self.terms:
                cxp = np.asarray(ct(Xp), dtype=float)         # (m,)
                blk = rbf_mixed_deriv(X, Xp, ds, dt, ls, sigma2)
                out += blk * cx[:, None] * cxp[None, :]
        return out
