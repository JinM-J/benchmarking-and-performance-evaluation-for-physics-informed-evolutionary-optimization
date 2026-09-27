# problems/f09.py
"""F09 (paper F9): parametric reaction-diffusion optimization.

    u_t = D*u_xx+alpha*u^2*(1-u), D=0.05, alpha=1.0
    (x,t,mu) in [-2,2] x [0,2] x [-1,1]
    u(x,0,mu)=sin(pi*(x+mu))+mu*sin(3*pi*mu*(x-mu))
States and first spatial derivatives are periodic in x.
Decision and query spaces both use (x,t,mu); mu is not differentiated.

The unconstrained objective is a product of two quartic polynomials in
u(x,t,mu) and J(mu); its formula first shifts J by +0.3.
J is the integral of u, approximated on a 10x10 left-endpoint Riemann grid
with dx=0.4 and dt=0.2. Constraint violation is identically zero.

Reference/sampling queries use periodic x mapping and clipping to data
bounds; optimizer fitness queries bypass this mapping. The fitness path
rounds surrogate inputs and outputs to float32 according to the protocol."""
import numpy as np

from problems.base import (PDEProblem, PhysicsSpec, InitialCondition, BoundaryCondition,
                           LinearOperator, LinearOpTerm)
from problems.regions import random_face_region


class ParamReactionDiffusionPhysics(PhysicsSpec):
    """Parametric reaction-diffusion PDE for F09."""

    def __init__(self, xmin, xmax, tmin, tmax, mu_min, mu_max, D, alpha):
        self.xmin, self.xmax = xmin, xmax
        self.tmin, self.tmax = tmin, tmax
        self.mu_min, self.mu_max = mu_min, mu_max
        self.D, self.alpha = float(D), float(alpha)
        self._qb = np.array([[xmin, xmax], [tmin, tmax], [mu_min, mu_max]],
                            dtype=float)

    @property
    def coordinate_names(self) -> tuple:
        return ("x", "t", "mu")

    @property
    def time_axis(self):
        return 1

    def required_derivatives(self) -> set:
        # mu is a parameter: evaluate it without differentiation.
        return {("t",), ("x", "x")}

    def residual(self, Q, u, d):
        """Residual: u_t-D*u_xx-alpha*u^2*(1-u)."""
        return (d[("t",)] - self.D * d[("x", "x")]
                - self.alpha * (u ** 2) * (1.0 - u))

    def linear_operator(self):
        """Exact linear part: u_t-D*u_xx=0.

        The derivative order on mu is zero. Residual noise absorbs
        alpha*u^2*(1-u)."""
        return LinearOperator(terms=(
            LinearOpTerm((0, 1, 0), 1.0),
            LinearOpTerm((2, 0, 0), -self.D),
        ))

    # Analytic initial values at t=0, with random (x,mu) collocation.
    def initial_conditions(self):
        return (
            InitialCondition(
                region=random_face_region(self._qb, axis=1, value=self.tmin,
                                          seed=20160),
                target=lambda Q: (np.sin(np.pi * (Q[:, 0] + Q[:, 2]))
                                  + Q[:, 2] * np.sin(3.0 * np.pi * Q[:, 2]
                                                     * (Q[:, 0] - Q[:, 2]))),
                derivative=(),
            ),
        )

    # Periodic boundary values and first spatial derivatives.
    def boundary_conditions(self):
        # Shared seeds pair the free (t,mu) coordinates pointwise.
        left = random_face_region(self._qb, axis=0, value=self.xmin, seed=90210)
        right = random_face_region(self._qb, axis=0, value=self.xmax, seed=90210)
        return (
            BoundaryCondition(
                kind="periodic_derivative",
                region_pair=(left, right),
                derivative=("x",),
            ),
        )


class F09(PDEProblem):
    """Parametric reaction-diffusion PDE with integral objective F(u,J(mu))."""

    name = "F09"

    # Round optimizer-to-surrogate inputs to float32.
    surrogate_query_dtype = np.float32

    def __init__(
        self,
        xmin: float = -2.0, xmax: float = 2.0,
        tmin: float = 0.0, tmax: float = 2.0,
        mu_min: float = -1.0, mu_max: float = 1.0,
        D: float = 0.05, alpha: float = 1.0,
        j_nx: int = 10, j_nt: int = 10,
    ):
        self.xmin, self.xmax = xmin, xmax
        self.tmin, self.tmax = tmin, tmax
        self.mu_min, self.mu_max = mu_min, mu_max
        self._physics = ParamReactionDiffusionPhysics(
            xmin, xmax, tmin, tmax, mu_min, mu_max, D, alpha
        )
        # J(mu) quadrature uses the full spatial and temporal domain.
        self.j_nx, self.j_nt = int(j_nx), int(j_nt)
        self._j_dx = (xmax - xmin) / self.j_nx
        self._j_dt = (tmax - tmin) / self.j_nt
        # Dataset grid parameters, populated by attach_data().
        self._x_min_data = self._x_max_data = self._dx_data = None
        self._t_min_data = self._t_max_data = None
        self._mu_min_data = self._mu_max_data = None
        self._j_xt = None

    # Decision and query spaces.
    @property
    def decision_bounds(self) -> np.ndarray:
        return np.array(
            [[self.xmin, self.xmax],
             [self.tmin, self.tmax],
             [self.mu_min, self.mu_max]],
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

    # Attach periodic mapping parameters and the J quadrature grid.
    def attach_data(self, data_path: str) -> None:
        data = np.load(data_path)
        xg = np.asarray(data["x"], dtype=float)
        tg = np.asarray(data["t"], dtype=float)
        mg = np.asarray(data["mu"], dtype=float)
        self._x_min_data, self._x_max_data = float(xg.min()), float(xg.max())
        self._dx_data = float(xg[1] - xg[0])
        self._t_min_data, self._t_max_data = float(tg.min()), float(tg.max())
        self._mu_min_data, self._mu_max_data = float(mg.min()), float(mg.max())

        # Left-endpoint Riemann grid: periodically map x, then store float32 coordinates.
        x_lin = np.linspace(self.xmin, self.xmax, self.j_nx, endpoint=False)
        t_lin = np.linspace(self.tmin, self.tmax, self.j_nt, endpoint=False)
        X, T = np.meshgrid(x_lin, t_lin, indexing="ij")
        xf = self._map_x(X.reshape(-1))
        tf = np.clip(T.reshape(-1), self._t_min_data, self._t_max_data)
        self._j_xt = np.stack(
            [xf.astype(np.float32), tf.astype(np.float32)], axis=1
        )

    # Evaluation axes use problem bounds, with endpoint=False for x.
    def metric_axes_3d(self, reference, nx: int, nt: int, nmu: int):
        x = np.linspace(self.xmin, self.xmax, nx, endpoint=False,
                        dtype=np.float32)
        t = np.linspace(self.tmin, self.tmax, nt, dtype=np.float32)
        mu = np.linspace(self.mu_min, self.mu_max, nmu, dtype=np.float32)
        return x, t, mu

    # Normalize queries: periodically map x and clip t/mu to data bounds.
    def _map_x(self, x: np.ndarray) -> np.ndarray:
        """Periodically map x to the reference data grid."""
        x = np.asarray(x, dtype=np.float64)
        Lx = float(self.xmax - self.xmin)
        xw = ((x - self.xmin) % Lx) + self.xmin
        mask = xw > self._x_max_data
        if np.any(mask):
            xw[mask] = xw[mask] - Lx + self._dx_data
        return np.clip(xw, self._x_min_data, self._x_max_data)

    def normalize_queries(self, Q: np.ndarray) -> np.ndarray:
        if self._dx_data is None:
            raise RuntimeError("F09: Call attach_data first; periodic mapping requires the dataset grid")
        Q = np.asarray(Q, dtype=np.float64)
        if Q.ndim == 1:
            Q = Q.reshape(1, -1)
        Q = Q.copy()
        Q[:, 0] = self._map_x(Q[:, 0])
        Q[:, 1] = np.clip(Q[:, 1], self._t_min_data, self._t_max_data)
        Q[:, 2] = np.clip(Q[:, 2], self._mu_min_data, self._mu_max_data)
        return Q

    # Reduced objective: query both local u and J(mu) through state_provider.
    @staticmethod
    def _F(u_val: float, j_val: float) -> float:
        """Product of two quartic polynomials, after shifting j by +0.3."""
        j_val = j_val + 0.3
        term1 = 1.0 + (u_val - j_val + 1.0) ** 2 * (
            19.0 - 14.0 * u_val + 3.0 * u_val ** 2
            + 14.0 * j_val - 6.0 * u_val * j_val + 3.0 * j_val ** 2
        )
        term2 = 30.0 + (2.0 * u_val + 3.0 * j_val) ** 2 * (
            18.0 - 32.0 * u_val + 12.0 * u_val ** 2
            - 48.0 * j_val + 36.0 * u_val * j_val + 27.0 * j_val ** 2
        )
        return float(term1 * term2)

    def evaluate_fitness(self, decision: np.ndarray, state_provider) -> float:
        if self._j_xt is None:
            raise RuntimeError("F09: Call attach_data first; the J quadrature grid requires reference data")
        d = np.asarray(decision, dtype=float).reshape(-1)
        mu = float(d[2])
        # One batch: the local objective query first, then all J quadrature points.
        N = self._j_xt.shape[0]
        Q = np.empty((N + 1, 3), dtype=np.float64)
        Q[0] = d
        Q[1:, 0:2] = self._j_xt.astype(np.float64)
        Q[1:, 2] = mu
        u_all = np.asarray(state_provider.evaluate(Q), dtype=float).reshape(-1)
        u_val = float(u_all[0])
        j_val = float(u_all[1:].sum() * self._j_dx * self._j_dt)
        return self._F(u_val, j_val)

    def violation(self, decision: np.ndarray, state_provider) -> float:
        # Unconstrained problem.
        return 0.0

    def constraint_components(self, decision, state_provider) -> list:
        return []   # Unconstrained: no constraint channels.

    def constraint_meta(self) -> list:
        return []   # Unconstrained: no constraint channels.