# evaluation/reference.py
"""Stored PDE reference fields and interpolation; mathematical definitions reside in problems/."""
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator


class ReferenceDataset:
    """Load reference u from NPZ and interpolate along its declared axes.

    transpose=True handles fields stored as (axis1, axis0), such as F07.
    nan_fill="median" fills a copy for interpolation, limiting NaN propagation
    from holes; the original u_grid is retained for finite-mask evaluation.
    nan_fill_value is the fallback if nanmedian is not finite."""

    def __init__(self, path, transpose: bool = False, nan_fill: str = None,
                 nan_fill_value: float = 0.0, coord_keys: tuple = None,
                 sort_axes: bool = True):
        self.path = Path(path)
        data = np.load(self.path)
        # Axis keys may be explicit or inferred: x,t with optional mu/lognu.
        if coord_keys is None:
            coord_keys = ("x", "t") + (
                ("mu",) if "mu" in data.files
                else (("lognu",) if "lognu" in data.files else ())
            )
        self.coord_keys = tuple(coord_keys)
        grids = [np.asarray(data[k], dtype=float) for k in self.coord_keys]
        # `NpzFile.__getitem__` materializes the whole member on every access.
        # Read the multi-GB u member once and retain its original dtype.
        # The interpolation array is promoted to float64.
        u_raw = data["u"]
        label_dtype = np.dtype(u_raw.dtype)
        u = np.asarray(u_raw, dtype=float)
        del u_raw
        if transpose:
            u = u.T
        # Reverse descending axes and the matching u dimensions together.
        if sort_axes:
            for j, g in enumerate(grids):
                if g.size >= 2 and g[0] > g[-1]:
                    grids[j] = g[::-1].copy()
                    u = np.flip(u, axis=j).copy()
        self.grids = grids
        self.x_grid = grids[0]
        self.t_grid = grids[1]
        self.mu_grid = grids[2] if len(grids) > 2 else None
        self.u_grid = u
        # Label precision follows the stored dtype (float32 for F09).
        # Pool objectives and violations use these rounded labels.
        self.label_dtype = label_dtype
        data.close()

        self.axes_min = np.array([float(g.min()) for g in grids])
        self.axes_max = np.array([float(g.max()) for g in grids])
        self.x_min, self.x_max = float(self.axes_min[0]), float(self.axes_max[0])
        self.t_min, self.t_max = float(self.axes_min[1]), float(self.axes_max[1])

        # Use linear interpolation, optionally filling NaNs first.
        u_fill = u
        if nan_fill == "median" and np.isnan(u).any():
            u_fill = u.copy()
            fill_val = np.nanmedian(u)
            if not np.isfinite(fill_val):
                fill_val = nan_fill_value
            u_fill[np.isnan(u_fill)] = fill_val

        self.interpolator = RegularGridInterpolator(tuple(grids), u_fill)

        if self.mu_grid is None:
            # Flatten the original grid in x-major, t-minor order for MSE/RMSE.
            self.eval_x_flat = np.repeat(self.x_grid, self.t_grid.size)
            self.eval_t_flat = np.tile(self.t_grid, self.x_grid.size)
            self.eval_u_flat = self.u_grid.reshape(-1)
        else:
            self.mu_min = float(self.mu_grid.min())
            self.mu_max = float(self.mu_grid.max())

    def query(self, Q: np.ndarray) -> np.ndarray:
        """Interpolate true u at query points, clipping coordinates to the stored domain."""
        Q = np.asarray(Q, dtype=float)
        if Q.ndim == 1:
            Q = Q.reshape(1, -1)
        Qc = np.clip(Q, self.axes_min, self.axes_max)
        return self.interpolator(Qc)
