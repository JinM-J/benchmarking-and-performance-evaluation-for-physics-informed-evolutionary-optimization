# surrogates/gp.py
"""GaussianProcessRegressor surrogate on query coordinates normalized to [0,1]^d.

Default kernel: RBF([0.2], bounds=(1e-3,1e2)) + WhiteKernel(1e-8).
Other defaults: alpha=1e-8, normalize_y=True, random_state=seed, and one
optimizer restart. Skip fitting when fewer than two samples are available.
Protocol overrides preserve problem-specific baseline settings.

Pass u to sklearn in its original pool dtype (float32). Promoting to float64
can change the final L-BFGS theta bits and prevent bitwise reproduction of reference runs.
"""
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RBF, WhiteKernel

from surrogates.base import Surrogate


class GPSurrogate(Surrogate):
    name = "GP"

    def __init__(self):
        self.gp = None
        self.qmin = self.qmax = None

    def setup(self, problem, seed: int, config: dict = None):
        config = config or {}
        qb = np.asarray(problem.query_bounds, dtype=float)
        self.qmin, self.qmax = qb[:, 0], qb[:, 1]
        # Protocol-specific normalization epsilon (default 0).
        # F01-F05/F08 use zero; F07 uses a denominator offset of 1e-12.
        self.norm_eps = float(config.get("normalize_eps", 0.0))

        bounds_config = config.get('length_scale_bounds', (0.001, 100.0))
        ls_bounds = (0.0, np.inf) if isinstance(bounds_config, str) and bounds_config == 'unbounded' else tuple(bounds_config)
        length_scale = config.get('length_scale', [0.2])
        if 'length_scale' in config:
            scales = np.asarray(length_scale, dtype=float)
            if scales.ndim > 1 or scales.size not in (1, qb.shape[0]) or (not np.isfinite(scales).all()) or np.any(scales <= 0.0):
                raise ValueError('GP length_scale must be positive and finite, with one entry or one per query coordinate')
        white = float(config.get("white_kernel", 1e-8))
        # Use sklearn default WhiteKernel noise bounds (1e-5,1e5) unless overridden.
        # F09 sets (1e-10,1e-5) explicitly through the protocol.
        self.center_y = config.get('center_y', True)
        if not isinstance(self.center_y, (bool, np.bool_)):
            raise ValueError('GP center_y must be a boolean')
        self.center_y = bool(self.center_y)
        self.target_scale = 1.0
        wb = config.get("white_kernel_bounds", None)
        white_kernel = (WhiteKernel(white, tuple(wb)) if wb is not None
                        else WhiteKernel(white))
        self.kernel_type = config.get('kernel_type', 'rbf')
        if self.kernel_type == 'rbf':
            signal_kernel = RBF(length_scale, length_scale_bounds=ls_bounds)
        elif self.kernel_type == 'matern32':
            signal_kernel = Matern(length_scale, length_scale_bounds=ls_bounds, nu=1.5)
        else:
            raise ValueError("GP kernel_type must be 'rbf' or 'matern32'")
        kernel = signal_kernel + white_kernel
        self.gp = GaussianProcessRegressor(kernel=kernel, alpha=float(config.get('alpha', 1e-08)), normalize_y=self.center_y, random_state=seed, n_restarts_optimizer=int(config.get('n_restarts_optimizer', 1)))
        # F09+ baseline rounds predictions to float32 for both fitness and MSE.
        self.predict_float32 = bool(config.get("predict_float32", False))
        # Optional deterministic training subset avoids cubic GP cost for large training pools.
        # None uses the full pool and matches the F01-F11 reference runs.
        self.max_train_points = config.get("max_train_points", None)
        self.seed = seed

    def _normalize(self, Q: np.ndarray) -> np.ndarray:
        """Normalize each coordinate as (q-qmin)/(qmax-qmin+eps)."""
        Q = np.asarray(Q, dtype=float)
        return (Q - self.qmin) / (self.qmax - self.qmin + self.norm_eps)

    def fit(self, Q: np.ndarray, u: np.ndarray, **ctx):
        X = self._normalize(Q)
        # Keep the original u dtype to preserve fitted hyperparameter bits.
        y = np.asarray(u).reshape(-1)
        if X.shape[0] < 2:
            return
        if self.max_train_points is not None and X.shape[0] > int(self.max_train_points):
            # Fixed-seed uniform subsampling gives the same subset for the same pool.
            rng = np.random.default_rng(20260821)
            idx = rng.choice(X.shape[0], int(self.max_train_points), replace=False)
            X, y = X[idx], y[idx]
        if not self.center_y:
            scale = np.std(y)
            self.target_scale = scale if scale != 0.0 else 1.0
            y = y / self.target_scale
        self.gp.fit(X, y)

    def predict(self, Q: np.ndarray) -> np.ndarray:
        X = self._normalize(np.asarray(Q, dtype=float).reshape(-1, self.qmin.size))
        out = self.gp.predict(X, return_std=False).reshape(-1)
        if not getattr(self, 'center_y', True):
            out = out * self.target_scale
        if self.predict_float32:
            out = out.astype(np.float32)
        return out
