# problems/f11.py
"""F11 (paper F11): parametric forced heat-equation optimization.

    u_t=D*u_xx+s(x,t,mu), D=0.05 (overridden by dataset metadata)
    s=(1+mu)*2*sin(pi*x)*cos(2*pi*t)
    (x,t,mu) in [-2,2] x [0,2] x [-1,1]
    u(x,0,mu)=0.5*sin(pi*x)
States and first spatial derivatives are periodic in x.

The unconstrained objective is Ackley-type with x1=32*u, x2=20*(J-0.5).
Input work J(mu)=integral(u*s dx dt) uses a 10x10 Riemann grid:
x includes endpoints (dx=4/9), while t excludes the endpoint (dt=0.2).
Constraint violation is zero.

The fitness path rounds surrogate inputs and outputs to float32, and
pool objectives round both u and J to float32. Sampling queries are clipped.
Fitness uses separate calls for one local point and 100 quadrature points."""
import numpy as np

from problems.base import (PDEProblem, PhysicsSpec, InitialCondition, BoundaryCondition,
                           LinearOperator, LinearOpTerm)
from problems.regions import random_face_region


class ParamForcedHeatPhysics(PhysicsSpec):
    """Parametric forced heat PDE; attach_data overrides D from the dataset."""

    def __init__(self, xmin, xmax, tmin, tmax, mu_min, mu_max, D):
        self.xmin, self.xmax = xmin, xmax
        self.tmin, self.tmax = tmin, tmax
        self.mu_min, self.mu_max = mu_min, mu_max
        self.D = float(D)
        self._qb = np.array([[xmin, xmax], [tmin, tmax], [mu_min, mu_max]],
                            dtype=float)

    @property
    def coordinate_names(self) -> tuple:
        return ("x", "t", "mu")

    @property
    def time_axis(self):
        return 1

    def required_derivatives(self) -> set:
        return {("t",), ("x", "x")}

    def residual(self, Q, u, d):
        """Torch residual u_t-D*u_xx-s(x,t,mu), with analytic source s."""
        import torch
        s = (1.0 + Q[:, 2:3]) * 2.0 * torch.sin(np.pi * Q[:, 0:1]) \
            * torch.cos(2.0 * np.pi * Q[:, 1:2])
        return d[("t",)] - self.D * d[("x", "x")] - s

    def linear_operator(self):
        """Complete linear operator: u_t-D*u_xx=s(x,t,mu).

        The parameter mu enters the known right-hand side without differentiation."""
        return LinearOperator(
            terms=(
                LinearOpTerm((0, 1, 0), 1.0),
                LinearOpTerm((2, 0, 0), -self.D),
            ),
            rhs=lambda Q: (1.0 + Q[:, 2]) * 2.0 * np.sin(np.pi * Q[:, 0])
            * np.cos(2.0 * np.pi * Q[:, 1]),
        )

    # At t=0, use random (x,mu) collocation with u=0.5*sin(pi*x).
    def initial_conditions(self):
        return (
            InitialCondition(
                region=random_face_region(self._qb, axis=1, value=self.tmin,
                                          seed=41000),
                target=lambda Q: 0.5 * np.sin(np.pi * Q[:, 0]),
                derivative=(),
            ),
        )

    # Periodic boundary values and first spatial derivatives.
    def boundary_conditions(self):
        # Shared seeds pair the free (t,mu) coordinates pointwise.
        left = random_face_region(self._qb, axis=0, value=self.xmin, seed=57700)
        right = random_face_region(self._qb, axis=0, value=self.xmax, seed=57700)
        return (
            BoundaryCondition(
                kind="periodic_derivative",
                region_pair=(left, right),
                derivative=("x",),
            ),
        )


class F11(PDEProblem):
    """Parametric forced heat PDE with an input-work integral and Ackley-type objective."""

    name = "F11"

    # Round optimizer-to-surrogate inputs to float32.
    surrogate_query_dtype = np.float32

    def __init__(
        self,
        xmin: float = -2.0, xmax: float = 2.0,
        tmin: float = 0.0, tmax: float = 2.0,
        mu_min: float = -1.0, mu_max: float = 1.0,
        D: float = 0.05,
        j_nx: int = 10, j_nt: int = 10,
    ):
        self.xmin, self.xmax = xmin, xmax
        self.tmin, self.tmax = tmin, tmax
        self.mu_min, self.mu_max = mu_min, mu_max
        self._physics = ParamForcedHeatPhysics(
            xmin, xmax, tmin, tmax, mu_min, mu_max, D
        )
        self.j_nx, self.j_nt = int(j_nx), int(j_nt)
        # x includes endpoints: dx=(x2-x1)/(nx-1); t excludes endpoint: dt=(t2-t1)/nt.
        self._j_dx = (xmax - xmin) / (self.j_nx - 1)
        self._j_dt = (tmax - tmin) / self.j_nt
        self._axes_min_data = self._axes_max_data = None
        self._j_xt = None
        self._j_sin_x = self._j_cos_t = None

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

    # Attach bounds, D, and the source-weighted J quadrature grid.
    def attach_data(self, data_path: str) -> None:
        from evaluation.reference import ReferenceDataset
        ref = ReferenceDataset(data_path)
        self._axes_min_data = ref.axes_min
        self._axes_max_data = ref.axes_max
        # Dataset D takes precedence, including float32 rounding of 0.05.
        data = np.load(data_path)
        if "D" in data.files:
            self._physics.D = float(np.asarray(data["D"]).reshape(-1)[0])

        x_lin = np.linspace(self.xmin, self.xmax, self.j_nx, endpoint=True)
        t_lin = np.linspace(self.tmin, self.tmax, self.j_nt, endpoint=False)
        X, T = np.meshgrid(x_lin, t_lin, indexing="ij")
        xf = np.clip(X.reshape(-1), self._axes_min_data[0], self._axes_max_data[0])
        tf = np.clip(T.reshape(-1), self._axes_min_data[1], self._axes_max_data[1])
        self._j_xt = np.stack(
            [xf.astype(np.float32), tf.astype(np.float32)], axis=1
        )
        # Evaluate source weights on the unclipped linspace coordinates.
        self._j_sin_x = np.sin(np.pi * x_lin)
        self._j_cos_t = np.cos(2.0 * np.pi * t_lin)

    # Normalize queries by clipping to data bounds.
    def normalize_queries(self, Q: np.ndarray) -> np.ndarray:
        if self._axes_min_data is None:
            raise RuntimeError("F11: Call attach_data first")
        Q = np.asarray(Q, dtype=np.float64)
        if Q.ndim == 1:
            Q = Q.reshape(1, -1)
        return np.clip(Q, self._axes_min_data, self._axes_max_data)

    # Evaluation axes use data bounds with all endpoints included.
    def metric_axes_3d(self, reference, nx: int, nt: int, nmu: int):
        if self._axes_min_data is None:
            raise RuntimeError("F11: Call attach_data first")
        x = np.linspace(self._axes_min_data[0], self._axes_max_data[0], nx,
                        dtype=np.float32)
        t = np.linspace(self._axes_min_data[1], self._axes_max_data[1], nt,
                        dtype=np.float32)
        m = np.linspace(self._axes_min_data[2], self._axes_max_data[2], nmu,
                        dtype=np.float32)
        return x, t, m

    # Objective.
    @staticmethod
    def _F(u_val: float, j_val: float) -> float:
        """Ackley-type objective evaluated in float64."""
        x1 = 32.0 * u_val
        x2 = 20.0 * (j_val - 0.5)
        term1 = -20.0 * np.exp(-0.2 * np.sqrt((x1 * x1 + x2 * x2) / 2.0))
        term2 = -np.exp((np.cos(2.0 * np.pi * x1) + np.cos(2.0 * np.pi * x2)) / 2.0)
        return float(term1 + term2 + 20.0 + float(np.e))

    def _work_j(self, u_on_grid: np.ndarray, mu: float) -> float:
        """Input work J=sum(u*s)*dx*dt, with s=(1+mu)*2*sin(pi*x)*cos(2*pi*t)."""
        U = np.asarray(u_on_grid, dtype=np.float64).reshape(self.j_nx, self.j_nt)
        sin_x = self._j_sin_x.reshape(self.j_nx, 1)
        cos_t = self._j_cos_t.reshape(1, self.j_nt)
        S = (1.0 + float(mu)) * 2.0 * (sin_x * cos_t)
        return float(np.sum(U * S)) * self._j_dx * self._j_dt

    def evaluate_fitness(self, decision: np.ndarray, state_provider) -> float:
        if self._j_xt is None:
            raise RuntimeError("F11: Call attach_data first; the J quadrature grid requires reference data")
        d = np.asarray(decision, dtype=float).reshape(-1)
        mu = float(d[2])
        # Separate surrogate calls for the local point and the J quadrature grid.
        u_val = float(np.asarray(
            state_provider.evaluate(d.reshape(1, 3)), dtype=float
        ).reshape(-1)[0])
        j_val = self._j_via(d, mu, state_provider)
        return self._F(u_val, j_val)

    def evaluate_fitness_for_pool(self, decision: np.ndarray, state_provider) -> float:
        """Pool objective arithmetic rounds J to float32 before evaluating F."""
        d = np.asarray(decision, dtype=float).reshape(-1)
        mu = float(d[2])
        u_val = float(np.asarray(
            state_provider.evaluate(d.reshape(1, 3)), dtype=float
        ).reshape(-1)[0])
        j_val = float(np.float32(self._j_via(d, mu, state_provider)))
        return self._F(u_val, j_val)

    def _j_via(self, d, mu, state_provider) -> float:
        N = self._j_xt.shape[0]
        Qj = np.empty((N, 3), dtype=np.float64)
        Qj[:, 0:2] = self._j_xt.astype(np.float64)
        Qj[:, 2] = mu
        uj = np.asarray(state_provider.evaluate(Qj), dtype=float).reshape(-1)
        return self._work_j(uj, mu)

    def violation(self, decision: np.ndarray, state_provider) -> float:
        # Unconstrained problem.
        return 0.0

    def constraint_components(self, decision, state_provider) -> list:
        return []   # Unconstrained: no constraint channels.

    def constraint_meta(self) -> list:
        return []   # Unconstrained: no constraint channels.