# evaluation/metrics.py
"""Fixed-grid monitoring MSE and original-grid final MSE/RMSE."""
import numpy as np


class MetricGrid:
    """Cache the fixed evaluation grid and reference labels.

    Two-dimensional grids use float64 (default 200 by 200). Parametric 3D grids
    use float32 with endpoint=False on x. Queries are normalized by the problem;
    reference interpolation is batched and labels are rounded to float32."""

    def __init__(self, problem, reference, nx_eval: int = 200, nt_eval: int = 200,
                 nmu_eval: int = None, batch_size: int = 65536):
        qb = np.asarray(problem.query_bounds, dtype=float)
        self.dim = qb.shape[0]
        self.batch_size = batch_size

        if self.dim == 2:
            x = np.linspace(qb[0, 0], qb[0, 1], nx_eval)
            t = np.linspace(qb[1, 0], qb[1, 1], nt_eval)
            X, T = np.meshgrid(x, t, indexing="xy")

            self.xf = X.reshape(-1)
            self.tf = T.reshape(-1)
            Q = np.c_[self.xf, self.tf]
            u_true = np.asarray(reference.query(Q), dtype=float).reshape(-1)

            # Exclude invalid queries (e.g. holes) and nonfinite reference values.
            valid = np.asarray(problem.is_valid_query(Q), dtype=bool) & np.isfinite(u_true)
            self.Q = Q[valid]
            self.u_true = u_true[valid]
        elif self.dim == 3:
            nmu_eval = int(nmu_eval) if nmu_eval is not None else nx_eval
            # Problem-defined axes determine domains and endpoint conventions.
            # Flatten in x-major, then t, then parameter order (meshgrid indexing='ij').
            x, t, mu = problem.metric_axes_3d(reference, nx_eval, nt_eval, nmu_eval)
            xf = np.repeat(x, nt_eval * nmu_eval)
            tf = np.tile(np.repeat(t, nmu_eval), nx_eval)
            muf = np.tile(mu, nx_eval * nt_eval)

            Q = np.stack([xf, tf, muf], axis=1).astype(np.float64)
            # Apply periodic mapping or clipping to the stored data domain.
            Q = np.asarray(problem.normalize_queries(Q), dtype=np.float64)
            self.Q = Q.astype(np.float32)   # Preserve float32 grid-cache precision.

            u_true = np.empty((self.Q.shape[0],), dtype=np.float32)
            for i0 in range(0, self.Q.shape[0], batch_size):
                i1 = min(self.Q.shape[0], i0 + batch_size)
                u_true[i0:i1] = np.asarray(
                    reference.query(self.Q[i0:i1].astype(np.float64)),
                    dtype=np.float32,
                ).reshape(-1)
            # Exclude invalid queries and nonfinite reference values.
            valid = np.asarray(problem.is_valid_query(self.Q), dtype=bool) \
                & np.isfinite(u_true)
            self.Q = self.Q[valid]
            self.u_true = u_true[valid]
        else:
            # For dim>3, use the problem's fixed evaluation set.
            # Label it once to avoid repeated physical solves at each snapshot.
            es = problem.metric_eval_set(reference)
            if es is None:
                raise ValueError(
                    f"A dim={self.dim} problem must implement metric_eval_set(reference)")
            self.Q = np.asarray(es[0], dtype=float)
            self.u_true = np.asarray(es[1], dtype=float)

    def mse(self, surrogate) -> float:
        """Surrogate MSE on the fixed grid or declared evaluation set."""
        if self.dim != 3:
            pred = surrogate.predict(self.Q)
            return float(np.mean((pred - self.u_true) ** 2))
        # Accumulate squared errors and counts in batches for 3D grids.
        sse, cnt = 0.0, 0
        N = self.Q.shape[0]
        for i0 in range(0, N, self.batch_size):
            i1 = min(N, i0 + self.batch_size)
            up = np.asarray(surrogate.predict(self.Q[i0:i1]),
                            dtype=np.float64).reshape(-1)
            ut = self.u_true[i0:i1].astype(np.float64, copy=False)
            err = up - ut
            sse += float(np.sum(err * err))
            cnt += int(err.size)
        return float(sse / cnt) if cnt > 0 else float("nan")


def mse_rmse_on_real_grid(surrogate, reference, batch_size: int = 65536):
    """Evaluate batched MSE/RMSE on the original NPZ grid, excluding NaN points."""
    Q = np.c_[reference.eval_x_flat, reference.eval_t_flat]
    u_true = reference.eval_u_flat

    mask = np.isfinite(u_true)
    if not np.all(mask):
        Q, u_true = Q[mask], u_true[mask]

    N = u_true.size
    if N < 1:
        return float("nan"), float("nan"), 0

    pred = np.empty(N, dtype=float)
    for i in range(0, N, batch_size):
        j = min(i + batch_size, N)
        pred[i:j] = surrogate.predict(Q[i:j])

    mse = float(np.mean((pred - u_true) ** 2))
    return mse, float(np.sqrt(mse)), int(N)
