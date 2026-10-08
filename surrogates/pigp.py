# surrogates/pigp.py
"""Operator-informed physics-informed Gaussian process (PIGP).

For linear L, u~GP(0,k) implies Lu~GP(0,L_x L_x' k). The joint covariance is
[[K_uu,K_uL],[K_Lu,K_LL]] (see kernels/physics_kernel.py), and targets combine
u observations with known rhs(Q) values at operator collocation points.
The rhs is analytic or interpolated from the F07 empirical field and costs no FE.
For nonlinear PDEs F01/F03/F08/F09/F10, use the exact linear part of each PDE;
learned likelihood noise_L absorbs omitted nonlinear terms rather than acting
as a manually chosen physics weight.

NumPy/SciPy Cholesky and L-BFGS-B support block covariance and noise. Coordinates
are normalized to [0,1]^d; derivatives use span_a^(-1) chain factors, while
operator coefficients are evaluated in physical coordinates. For
u_tilde=(u-m_u)/s_u, the operator target is (rhs-m_u*c_id)/s_u, where c_id sums
zeroth-order coefficients. Both blocks share s_u.

Hyperparameters are learned in two stages. Joint likelihoods with rhs=0 can
favor a degenerate constant-field/noise solution as length scales grow and
signal/residual noise shrink. Stage A learns length scales, sigma2 and noise_u
using only the data-block marginal likelihood. Stage B fixes those parameters
and learns noise_L from the joint likelihood. All optimization uses log
parameters, numerical gradients and deterministic initial values.
"""
import numpy as np
from scipy.optimize import minimize

from surrogates.base import Surrogate
from surrogates.kernels.base import rbf, rbf_mixed_deriv
from surrogates.kernels.physics_kernel import OperatorKernel


def normalize_operator(lin_op, qmin, qmax):
    """Convert a physical LinearOperator to normalized coordinates.

    An order p_a derivative contributes span_a^(-p_a), while coefficients are
    evaluated at physical coordinates. Return (terms_norm,c_id_fn): normalized
    terms for OperatorKernel and the summed zeroth-order coefficient function
    c_id_fn(Q_phys)->(N,) used to standardize targets.
    """
    span = qmax - qmin
    terms_norm = []
    id_coefs = []
    for term in lin_op.terms:
        scale = float(np.prod(span ** (-np.asarray(term.deriv, dtype=float))))
        coef = term.coef
        if callable(coef):
            def coef_fn(Xn, _c=coef, _s=scale):
                Q = Xn * span + qmin          # Evaluate coefficients at physical coordinates.
                return _s * np.asarray(_c(Q), dtype=float)
        else:
            cval = float(coef) * scale
            def coef_fn(Xn, _v=cval):
                return np.full(Xn.shape[0], _v)
        terms_norm.append((tuple(term.deriv), coef_fn))
        if all(o == 0 for o in term.deriv):
            id_coefs.append(term.coef)

    def c_id_fn(Q):
        if not id_coefs:
            return np.zeros(Q.shape[0])
        out = np.zeros(Q.shape[0])
        for c in id_coefs:
            out = out + (np.asarray(c(Q), dtype=float) if callable(c)
                         else float(c))
        return out

    return terms_norm, c_id_fn


class PIGPSurrogate(Surrogate):
    """Operator-conditioned GP with setup/fit/predict methods."""

    name = "PIGP"

    def __init__(self):
        self.kernel = None
        self.qmin = self.qmax = None

    def setup(self, problem, seed: int, config: dict = None):
        config = config or {}
        center_y = config.get("center_y", True)
        if not isinstance(center_y, (bool, np.bool_)):
            raise ValueError("PIGP center_y must be a boolean")
        self.center_y = bool(center_y)
        qb = np.asarray(problem.query_bounds, dtype=float)
        self.qmin, self.qmax = qb[:, 0], qb[:, 1]
        self.problem = problem

        lin_op = problem.physics.linear_operator()
        if lin_op is None:
            raise ValueError(
                f"{problem.name} does not provide linear_operator; cannot construct PIGP"
                f" (no fallback to a data-only GP)"
            )
        terms_norm, self._c_id_fn = normalize_operator(lin_op, self.qmin,
                                                       self.qmax)
        self.kernel = OperatorKernel(terms_norm)
        self._rhs = lin_op.rhs

        # Fixed-seed setup collocation; analytic/interpolated rhs labels cost no FE.
        # Default uniform sampling with rejection preserves F01-F06 behavior.
        # Select problem_interior explicitly for categorical axes or singular margins.
        m = int(config.get("n_collocation", 100))
        sampler = config.get("collocation_sampler", "uniform")
        if sampler == "problem_interior":
            Q_L = problem.sample_interior_queries(m, seed)
            if Q_L is None:
                raise ValueError(
                    f"{problem.name} does not provide sample_interior_queries; "
                    "cannot use collocation_sampler=problem_interior"
                )
            Q_L = np.asarray(Q_L, dtype=float)
            if Q_L.shape != (m, qb.shape[0]):
                raise ValueError(
                    f"{problem.name} interior collocation shape is {Q_L.shape}; "
                    f"expected {(m, qb.shape[0])}"
                )
            in_bounds = np.all((Q_L >= self.qmin) & (Q_L <= self.qmax))
            valid = np.asarray(problem.is_valid_query(Q_L), dtype=bool)
            if (not np.isfinite(Q_L).all()) or (not in_bounds) or (not valid.all()):
                raise ValueError(f"{problem.name} sample_interior_queries returned invalid collocation points")
            self._Q_L = Q_L
        elif sampler == "uniform":
            # Reject and resample invalid F07 points inside holes.
            rng = np.random.default_rng(seed)
            pts = []
            attempts = 0
            while len(pts) < m:
                attempts += 1
                if attempts > 100:
                    raise RuntimeError(
                        f"PIGP could not sample {m} collocation points after 100 rejection rounds: only "
                        f"{len(pts)} found; check is_valid_query and the query domain"
                    )
                cand = rng.uniform(self.qmin, self.qmax,
                                   (max(2 * m, 32), qb.shape[0]))
                ok = problem.is_valid_query(cand)
                pts.extend(map(tuple, cand[ok]))
            self._Q_L = np.asarray(pts[:m])
        else:
            raise KeyError(
                f"Unknown PIGP collocation_sampler={sampler!r}; "
                "supported: uniform, problem_interior"
            )
        r = (self._rhs(self._Q_L) if self._rhs is not None
             else np.zeros(self._Q_L.shape[0]))
        self._r_L = np.asarray(r, dtype=float).reshape(-1)

        # Log-space initialization/bounds use length_scale_bounds and white_kernel.
        # The obsolete transform_a/b keys are ignored.
        dim = qb.shape[0]
        ls0 = config.get("kernel_length_scale", None) or [0.2] * dim
        self._ls0 = np.asarray(ls0, dtype=float)
        ls_b = tuple(config.get("length_scale_bounds", (1e-3, 1e2)))
        # Use protocol white_kernel_bounds for noise_u when supplied (F09-F11).
        # Otherwise use the default wide bounds.
        nu_b = tuple(config.get("white_kernel_bounds", (1e-10, 1e2)))
        self._theta0 = np.concatenate([
            np.log(self._ls0), [np.log(1.0)],          # ℓ_a, σ²
            [np.log(float(config.get("white_kernel", 1e-5)))],   # noise_u
            [np.log(float(config.get("collocation_noise", 1e-2)))],  # noise_L
        ])
        self._bounds = ([(np.log(ls_b[0]), np.log(ls_b[1]))] * dim
                        + [(np.log(1e-4), np.log(1e4))]          # σ²
                        + [(np.log(nu_b[0]), np.log(nu_b[1]))]   # noise_u
                        # The noise_L floor 1e-8 guards against a nearly singular residual block;
                        # it is not the primary mechanism for model-mismatch control.
                        + [(np.log(1e-8), np.log(1e2))])         # noise_L
        self._n_restarts = int(config.get("n_restarts_optimizer", 1))
        self.predict_float32 = bool(config.get("predict_float32", False))
        # Joint Cholesky costs O(n^3); large training pools may use a deterministic subset.
        # Subsampling requires an explicit protocol cap; None preserves the full pool.
        self.max_train_points = config.get("max_train_points", None)

        self._fitted = False

    def _normalize(self, Q):
        Q = np.asarray(Q, dtype=float)
        return (Q - self.qmin) / (self.qmax - self.qmin)

    def _split_theta(self, theta):
        dim = self.qmin.size
        ls = np.exp(theta[:dim])
        sigma2 = float(np.exp(theta[dim]))
        noise_u = float(np.exp(theta[dim + 1]))
        noise_l = float(np.exp(theta[dim + 2]))
        return ls, sigma2, noise_u, noise_l

    def _joint_cov(self, Xu, ls, sigma2, noise_u, noise_l):
        """Joint covariance with separate block-noise diagonals."""
        XL = self._XL
        Kuu = self.kernel.k_uu(Xu, Xu, ls, sigma2)
        KuL = self.kernel.k_uL(Xu, XL, ls, sigma2)
        KLL = self.kernel.k_LL(XL, XL, ls, sigma2)
        n, m = Xu.shape[0], XL.shape[0]
        K = np.empty((n + m, n + m))
        K[:n, :n] = Kuu
        K[:n, n:] = KuL
        K[n:, :n] = KuL.T
        K[n:, n:] = KLL
        K[np.arange(n), np.arange(n)] += noise_u
        K[n + np.arange(m), n + np.arange(m)] += noise_l
        return K

    def _nll_data(self, theta3, t_u):
        """Stage A: data-only negative log marginal likelihood."""
        dim = self.qmin.size
        ls = np.exp(theta3[:dim])
        sigma2 = float(np.exp(theta3[dim]))
        noise_u = float(np.exp(theta3[dim + 1]))
        K = rbf(self._Xu, self._Xu, ls, sigma2)
        K[np.diag_indices_from(K)] += noise_u
        n = K.shape[0]
        jitter = 1e-10
        for _ in range(7):
            try:
                L = np.linalg.cholesky(K + jitter * np.eye(n))
                break
            except np.linalg.LinAlgError:
                jitter *= 10.0
        else:
            return 1e12
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, t_u))
        return float(0.5 * t_u @ alpha + np.log(np.diag(L)).sum()
                     + 0.5 * n * np.log(2.0 * np.pi))

    def _nll_joint_noisel(self, log_nl, y, ls, sigma2, noise_u):
        """Stage B: joint negative log likelihood in log noise_L, with fixed kernel parameters."""
        K = self._joint_cov(self._Xu, ls, sigma2, noise_u,
                            float(np.exp(log_nl[0])))
        n = K.shape[0]
        jitter = 1e-10
        for _ in range(7):
            try:
                L = np.linalg.cholesky(K + jitter * np.eye(n))
                break
            except np.linalg.LinAlgError:
                jitter *= 10.0
        else:
            return 1e12
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
        return float(0.5 * y @ alpha + np.log(np.diag(L)).sum()
                     + 0.5 * n * np.log(2.0 * np.pi))

    def fit(self, Q: np.ndarray, u: np.ndarray, **ctx):
        Q = np.asarray(Q, dtype=float).reshape(-1, self.qmin.size)
        y_u = np.asarray(u, dtype=float).reshape(-1)
        if Q.shape[0] < 2:
            return   # Skip fitting when too few samples are available.
        if self.max_train_points is not None and Q.shape[0] > int(self.max_train_points):
            # Use the same fixed subsampling seed as other field surrogates.
            rng = np.random.default_rng(20260821)
            idx = rng.choice(Q.shape[0], int(self.max_train_points), replace=False)
            Q, y_u = Q[idx], y_u[idx]
        X = self._normalize(Q)

        # Standardize both blocks with the shared scale s_u.
        self._m_u = float(y_u.mean()) if self.center_y else 0.0
        self._s_u = max(float(y_u.std()), 1e-12)
        t_u = (y_u - self._m_u) / self._s_u
        t_l = (self._r_L - self._m_u * self._c_id_fn(self._Q_L)) / self._s_u

        self._Xu = X
        self._XL = self._normalize(self._Q_L)
        y = np.concatenate([t_u, t_l])

        dim = self.qmin.size
        # Stage A: learn length scales, sigma2 and noise_u from the data block.
        theta3_0 = self._theta0[:dim + 2]
        bounds3 = self._bounds[:dim + 2]
        # n_restarts=0 optimizes the initial point once; positive values add deterministic starts.
        best_f, best3 = np.inf, theta3_0
        for r in range(max(1, self._n_restarts)):
            t0 = theta3_0 if r == 0 else theta3_0 + 0.1 * r
            res = minimize(self._nll_data, t0, args=(t_u,),
                           method="L-BFGS-B", bounds=bounds3)
            if res.fun < best_f:
                best_f, best3 = res.fun, res.x
        ls = np.exp(best3[:dim])
        sigma2 = float(np.exp(best3[dim]))
        noise_u = float(np.exp(best3[dim + 1]))

        # Stage B: optimize only noise_L in the joint likelihood.
        res_b = minimize(self._nll_joint_noisel,
                         np.array([self._theta0[dim + 2]]),
                         args=(y, ls, sigma2, noise_u),
                         method="L-BFGS-B",
                         bounds=[self._bounds[dim + 2]])
        noise_l = float(np.exp(res_b.x[0]))
        self._theta = np.concatenate([np.log(ls), [np.log(sigma2)],
                                      [np.log(noise_u)], [np.log(noise_l)]])

        K = self._joint_cov(X, ls, sigma2, noise_u, noise_l)
        jitter = 1e-10
        n_jitter = 0
        for _ in range(7):
            try:
                L = np.linalg.cholesky(K + jitter * np.eye(K.shape[0]))
                break
            except np.linalg.LinAlgError:
                jitter *= 10.0
                n_jitter += 1
        else:
            raise np.linalg.LinAlgError(
                "PIGP joint covariance Cholesky failed with jitter up to 1e-4; "
                "check the operator and noise configuration"
            )
        self._alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
        self._fitted = True

        # Condition diagnostics are separate from ranking metrics; high-order K_LL can be ill-conditioned.
        # Estimate reciprocal condition number from Cholesky with dpocon in O(n^2).
        from scipy.linalg.lapack import dpocon
        anorm = float(np.linalg.norm(K, 1))
        # np.linalg.cholesky returns lower-triangular L; dpocon selects by uplo.
        # Transpose L and pass uplo=U for the equivalent upper-triangular factor.
        rcond = float(dpocon(np.asfortranarray(L.T), anorm, uplo=b'U')[0])
        self.last_fit_diagnostics = {"jitter": n_jitter, "rcond": rcond}

    def predict_with_derivatives(self, Q: np.ndarray):
        """Posterior mean and physical derivatives in float64, without predict_float32 rounding.

        Return (u,d), where u has shape (N,) and d maps axis-name tuples from
        physics.required_derivatives() to arrays of shape (N,).
        E[d^p u(x*)] = [d_x^p k(x*,X_u), d_x^p L_x' k(x*,X_L)] alpha.
        Multiply normalized derivatives by s_u*prod_a span_a^(-p_a);
        the constant m_u vanishes. Supports full and linear residual metrics.
        """
        if not self._fitted:
            raise RuntimeError("PIGP must be fitted before predicting derivatives")
        Q = np.asarray(Q, dtype=float).reshape(-1, self.qmin.size)
        X = self._normalize(Q)
        names = self.problem.physics.coordinate_names
        span = self.qmax - self.qmin
        ls, sigma2, _, _ = self._split_theta(self._theta)

        ks_u = rbf(X, self._Xu, ls, sigma2)
        ks_l = self.kernel.k_uL(X, self._XL, ls, sigma2)
        u = (np.concatenate([ks_u, ks_l], axis=1) @ self._alpha) \
            * self._s_u + self._m_u

        d = {}
        zero = tuple(0 for _ in names)
        for key in self.problem.physics.required_derivatives():
            # Convert an axis-name sequence to per-axis derivative orders.
            p = tuple(key.count(nm) for nm in names)
            blk_u = rbf_mixed_deriv(X, self._Xu, p, zero, ls, sigma2)
            blk_l = self.kernel.k_duL(X, self._XL, p, ls, sigma2)
            scale = self._s_u * float(np.prod(span ** (-np.asarray(p, float))))
            d[key] = (np.concatenate([blk_u, blk_l], axis=1)
                      @ self._alpha) * scale
        return np.asarray(u, dtype=float).reshape(-1), d

    def predict(self, Q: np.ndarray) -> np.ndarray:
        X = self._normalize(np.asarray(Q, dtype=float)
                            .reshape(-1, self.qmin.size))
        ls, sigma2, _, _ = self._split_theta(self._theta)
        ks_u = rbf(X, self._Xu, ls, sigma2)                 # k(x*, X_u)
        ks_l = self.kernel.k_uL(X, self._XL, ls, sigma2)    # L_{x'} k(x*, X_L)
        mu = np.concatenate([ks_u, ks_l], axis=1) @ self._alpha
        out = mu * self._s_u + self._m_u
        if self.predict_float32:
            out = out.astype(np.float32)
        return np.asarray(out, dtype=float).reshape(-1)
