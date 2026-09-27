# problems/f07.py
"""F07 (paper F7): steady Poisson optimization on a perforated domain.

    u_x1x1 + u_x2x2 = f(x1,x2), (x1,x2) in [0,5]^2 minus four circular holes
    Outer edges: u=0.2; hole walls: u=1.0.
The source is interpolated from the reference dataset through attach_data.
There is no time axis or initial condition.

Decision and query coordinates are (x1,x2). Points inside holes are invalid;
fitness evaluation uses the penalty state nan_penalty_u=1e6 there.
The unconstrained Branin-type objective is
    f(x1,u) = (u-5.1*x1^2/(4*pi^2)+5*x1/pi-6)^2
              +10*(1-1/(8*pi))*cos(x1)+10.
Constraint violation is identically zero."""
import numpy as np
import torch
from scipy.interpolate import RegularGridInterpolator

from problems.base import (PDEProblem, PhysicsSpec, BoundaryCondition,
                           LinearOperator, LinearOpTerm)
from problems.regions import face_region, circle_boundary, union_region


class PerforatedPoissonPhysics(PhysicsSpec):
    """Steady Poisson PDE on the F07 perforated domain."""

    def __init__(self, xmin, xmax, tmin, tmax, holes, u_outer, u_obs):
        self.xmin, self.xmax = xmin, xmax
        self.tmin, self.tmax = tmin, tmax
        self.holes = tuple(holes)
        self.u_outer = float(u_outer)
        self.u_obs = float(u_obs)
        self._qb = np.array([[xmin, xmax], [tmin, tmax]], dtype=float)
        self._f_interp = None   # Source interpolator attached by set_source().

    @property
    def coordinate_names(self) -> tuple:
        return ("x1", "x2")

    @property
    def time_axis(self):
        return None   # Steady equation: no time axis.

    def required_derivatives(self) -> set:
        # Poisson：u_x1x1 + u_x2x2 = f
        return {("x1", "x1"), ("x2", "x2")}

    # Source field loaded from reference data.
    def set_source(self, f_interp):
        """Attach the source interpolator, called by problem.attach_data()."""
        self._f_interp = f_interp

    def residual(self, Q, u, d):
        """Residual u_x1x1+u_x2x2-f, with interpolated source field f."""
        if self._f_interp is None:
            raise RuntimeError(
                "F07 source f is unavailable; call problem.attach_data(data_path) first"
            )
        pts = Q.detach().cpu().numpy()
        fv = self._f_interp(np.c_[pts[:, 0], pts[:, 1]]).reshape(-1, 1)
        fv = np.where(np.isfinite(fv), fv, 0.0)
        f_t = torch.tensor(fv, dtype=torch.float32, device=Q.device)
        return d[("x1", "x1")] + d[("x2", "x2")] - f_t

    def _rhs_np(self, Q):
        """NumPy source evaluation for the PIGP right-hand side.

        Share the residual interpolator and finite-value handling:
        NaNs inside holes or outside the domain map to zero."""
        if self._f_interp is None:
            raise RuntimeError(
                "F07 source f is unavailable; call problem.attach_data(data_path) first"
            )
        fv = self._f_interp(np.c_[Q[:, 0], Q[:, 1]]).reshape(-1)
        return np.where(np.isfinite(fv), fv, 0.0)

    def linear_operator(self):
        """Complete Poisson operator: u_x1x1+u_x2x2=f, with interpolated source."""
        return LinearOperator(
            terms=(
                LinearOpTerm((2, 0), 1.0),
                LinearOpTerm((0, 2), 1.0),
            ),
            rhs=self._rhs_np,
        )

    # No initial condition for this steady equation.
    def initial_conditions(self):
        return ()

    # Outer boundaries u=0.2; union of hole walls u=1.0.
    def boundary_conditions(self):
        qb = self._qb
        u_out = self.u_outer
        bcs = tuple(
            BoundaryCondition(
                kind="dirichlet",
                region=face_region(qb, axis=ax, value=val, sample_axis=sax),
                target=lambda Q, v=u_out: np.full(Q.shape[0], v),
            )
            for ax, val, sax in (
                (0, self.xmin, 1), (0, self.xmax, 1),   # Left/right edges, sampled along x2.
                (1, self.tmin, 0), (1, self.tmax, 0),   # Bottom/top edges, sampled along x1.
            )
        )
        # Allocate hole-wall points in proportion to the four circumferences.
        # Use one union boundary loss with the protocol obs_bc_points budget.
        hole_regions = [circle_boundary(cx, ct, r) for (cx, ct, r) in self.holes]
        perims = np.array([2.0 * np.pi * r for (_, _, r) in self.holes])
        holes_bc = BoundaryCondition(
            kind="dirichlet",
            region=union_region(hole_regions, weights=perims, min_per=50),
            target=lambda Q: np.full(Q.shape[0], self.u_obs),
            points_key="obs_bc_points",
        )
        return bcs + (holes_bc,)


class F07(PDEProblem):
    """Perforated-domain Poisson PDE with an unconstrained Branin-type objective."""

    name = "F07"

    HOLES = (
        (4.3, 3.5, 0.50),
        (1.5, 1.2, 0.40),
        (2.2, 2.7, 0.60),
        (0.6, 0.5, 0.30),
    )

    def __init__(
        self,
        xmin: float = 0.0,
        xmax: float = 5.0,
        tmin: float = 0.0,
        tmax: float = 5.0,
        holes: tuple = HOLES,
        u_outer: float = 0.2,
        u_obs: float = 1.0,
        boundary_eps: float = 2e-3,
        nan_penalty_u: float = 1e6,
    ):
        self.xmin, self.xmax = xmin, xmax
        self.tmin, self.tmax = tmin, tmax
        self.holes = tuple(holes)
        self.u_outer = u_outer
        self.u_obs = u_obs
        # Sampling margin and penalty state for invalid or nonfinite queries.
        self.boundary_eps = boundary_eps
        self.nan_penalty_u = nan_penalty_u
        self._physics = PerforatedPoissonPhysics(
            xmin, xmax, tmin, tmax, self.holes, u_outer, u_obs
        )

    # Two-dimensional decision and query spaces coincide.
    @property
    def decision_bounds(self) -> np.ndarray:
        return np.array(
            [[self.xmin, self.xmax],
             [self.tmin, self.tmax]],
            dtype=np.float64,
        )

    @property
    def query_bounds(self) -> np.ndarray:
        return self.decision_bounds

    @property
    def physics(self) -> PhysicsSpec:
        return self._physics

    def decision_to_query(self, decision: np.ndarray) -> np.ndarray:
        return np.asarray(decision, dtype=float)

    # Points inside holes are invalid.
    def is_valid_query(self, Q: np.ndarray) -> np.ndarray:
        Q = np.asarray(Q, dtype=float).reshape(-1, 2)
        m = np.zeros(Q.shape[0], dtype=bool)
        for cx, ct, r in self.holes:
            m |= (Q[:, 0] - cx) ** 2 + (Q[:, 1] - ct) ** 2 <= r ** 2
        return ~m

    # Candidate sampling with boundary margins and interior rejection sampling.
    def sample_query_candidates(self, n: int, rng) -> np.ndarray:
        """Sample uniformly by axis with inward boundary_eps margins; two RNG calls."""
        qb = np.asarray(self.query_bounds, dtype=float)
        e = self.boundary_eps
        cols = [rng.uniform(qb[j, 0] + e, qb[j, 1] - e, size=int(n))
                for j in range(qb.shape[0])]
        return np.column_stack(cols)

    def sample_interior_queries(self, n: int, seed: int):
        """Sample exactly n interior PINN points by rejecting hole interiors.

        Candidate batch size is max(4*need,2048)."""
        rng = np.random.default_rng(seed)
        need = int(n)
        out = []
        while need > 0:
            m = max(need * 4, 2048)
            cand = self.sample_query_candidates(m, rng)
            ok = self.is_valid_query(cand)
            take = min(need, int(ok.sum()))
            if take > 0:
                out.append(cand[ok][:take])
                need -= take
        return np.concatenate(out, axis=0)

    # Attach source data after transposing to (x1,x2) order.
    def attach_data(self, data_path: str) -> None:
        data = np.load(data_path)
        x = np.asarray(data["x"], dtype=float)
        t = np.asarray(data["t"], dtype=float)
        f = np.asarray(data["f"], dtype=float)
        self._physics.set_source(RegularGridInterpolator(
            (x, t), f.T, bounds_error=False, fill_value=0.0
        ))

    def load_reference(self, data_path: str):
        """Load F07 states stored in (t,x) order and transpose them.

        Fill hole NaNs with nanmedian (falling back to u_outer) to prevent
        NaN contamination of neighboring interpolation cells."""
        from evaluation.reference import ReferenceDataset
        return ReferenceDataset(
            data_path, transpose=True, nan_fill="median",
            nan_fill_value=float(self.u_outer),
        )

    # Reduced objective; invalid/nonfinite queries use the penalty state.
    def evaluate_fitness(self, decision: np.ndarray, state_provider) -> float:
        Q = self.decision_to_query(decision).reshape(1, -1)
        if not bool(self.is_valid_query(Q)[0]):
            u = float(self.nan_penalty_u)
        else:
            u = float(state_provider.evaluate(Q)[0])
            if not np.isfinite(u):
                u = float(self.nan_penalty_u)
        x = float(Q[0, 0])
        # Branin-type objective with z1=x and z2=u.
        Z = (
            (u - (5.1 / (4.0 * np.pi ** 2)) * x ** 2 + (5.0 / np.pi) * x - 6.0) ** 2
            + 10.0 * (1.0 - (1.0 / (8.0 * np.pi))) * np.cos(x)
            + 10.0
        )
        return float(Z)

    def violation(self, decision: np.ndarray, state_provider) -> float:
        # Unconstrained problem.
        return 0.0

    def constraint_components(self, decision, state_provider) -> list:
        return []   # Unconstrained: no constraint channels.

    def constraint_meta(self) -> list:
        return []   # Unconstrained: no constraint channels.