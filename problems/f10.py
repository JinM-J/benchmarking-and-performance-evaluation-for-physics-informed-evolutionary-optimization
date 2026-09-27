# problems/f10.py
"""F10 (paper F10): parametric Burgers optimization.

    u_t+u*u_x=nu*u_xx, nu=exp(lognu)
    (x,t,lognu) in [-1,1] x [0,1] x [ln(1e-3),ln(1)]
    u(x,0,lognu)=-sin(pi*x)
    u(-1,t,lognu)=u(1,t,lognu)=0
The two boundaries share random (t,lognu) collocation coordinates.

The unconstrained dissipation-integral objective is
    J(lognu)=nu*integral(u_x^2 dx dt)
    F(u,J)=100*(u+0.6-(J+0.6)^2)^2+(J-0.4)^2.
J uses central spatial differences and a 10x10 Riemann grid: x includes
both endpoints (dx=2/9), while t excludes the endpoint (dt=0.1).

The fitness path rounds surrogate queries and predictions to float32.
Pool objectives also round u and J to float32. Sampling queries are clipped
without periodic mapping. Fitness evaluation makes two surrogate calls:
one local query and 100 quadrature queries. Constraint violation is zero."""
import numpy as np

from problems.base import (PDEProblem, PhysicsSpec, InitialCondition, BoundaryCondition,
                           LinearOperator, LinearOpTerm)
from problems.regions import random_face_region


class ParamBurgersPhysics(PhysicsSpec):
    """Parametric Burgers PDE with viscosity encoded by lognu."""

    def __init__(self, xmin, xmax, tmin, tmax, lognu_min, lognu_max):
        self.xmin, self.xmax = xmin, xmax
        self.tmin, self.tmax = tmin, tmax
        self.lognu_min, self.lognu_max = lognu_min, lognu_max
        self._qb = np.array([[xmin, xmax], [tmin, tmax],
                             [lognu_min, lognu_max]], dtype=float)

    @property
    def coordinate_names(self) -> tuple:
        return ("x", "t", "lognu")

    @property
    def time_axis(self):
        return 1

    def required_derivatives(self) -> set:
        return {("t",), ("x",), ("x", "x")}

    def residual(self, Q, u, d):
        """Torch residual: u_t+u*u_x-exp(lognu)*u_xx."""
        import torch
        nu = torch.exp(Q[:, 2:3])
        return d[("t",)] + u * d[("x",)] - nu * d[("x", "x")]

    def linear_operator(self):
        """Exact linear part: u_t-nu*u_xx=0, with nu=exp(lognu).

        The coefficient uses exp(Q[:,2]) without clipping, as in residual().
        Residual noise absorbs u*u_x."""
        return LinearOperator(terms=(
            LinearOpTerm((0, 1, 0), 1.0),
            LinearOpTerm((2, 0, 0), lambda Q: -np.exp(Q[:, 2])),
        ))

    # At t=0, use random (x,lognu) collocation with u=-sin(pi*x).
    def initial_conditions(self):
        return (
            InitialCondition(
                region=random_face_region(self._qb, axis=1, value=self.tmin,
                                          seed=31000),
                target=lambda Q: -np.sin(np.pi * Q[:, 0]),
                derivative=(),
            ),
        )

    # Zero Dirichlet values at x=+/-1 with shared (t,lognu) collocation.
    def boundary_conditions(self):
        # Shared seeds give matching free (t,lognu) coordinates on both faces.
        left = random_face_region(self._qb, axis=0, value=self.xmin, seed=47700)
        right = random_face_region(self._qb, axis=0, value=self.xmax, seed=47700)
        zero = lambda Q: np.zeros(Q.shape[0])
        return (
            BoundaryCondition(kind="dirichlet", region=left, target=zero),
            BoundaryCondition(kind="dirichlet", region=right, target=zero),
        )


class F10(PDEProblem):
    """Parametric Burgers PDE with a dissipation-integral objective."""

    name = "F10"

    # Round optimizer-to-surrogate inputs to float32.
    surrogate_query_dtype = np.float32

    def __init__(
        self,
        xmin: float = -1.0, xmax: float = 1.0,
        tmin: float = 0.0, tmax: float = 1.0,
        nu_min: float = 1e-3, nu_max: float = 1.0,
        j_nx: int = 10, j_nt: int = 10,
    ):
        self.xmin, self.xmax = xmin, xmax
        self.tmin, self.tmax = tmin, tmax
        self.lognu_min = float(np.log(nu_min))
        self.lognu_max = float(np.log(nu_max))
        self._physics = ParamBurgersPhysics(
            xmin, xmax, tmin, tmax, self.lognu_min, self.lognu_max
        )
        self.j_nx, self.j_nt = int(j_nx), int(j_nt)
        # x includes endpoints: dx=(x2-x1)/(nx-1); t excludes endpoint: dt=(t2-t1)/nt.
        self._j_dx = (xmax - xmin) / (self.j_nx - 1)
        self._j_dt = (tmax - tmin) / self.j_nt
        self._axes_min_data = self._axes_max_data = None
        self._j_xt = None

    # Decision and query spaces.
    @property
    def decision_bounds(self) -> np.ndarray:
        return np.array(
            [[self.xmin, self.xmax],
             [self.tmin, self.tmax],
             [self.lognu_min, self.lognu_max]],
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

    # Attach data bounds and the J quadrature grid.
    def attach_data(self, data_path: str) -> None:
        from evaluation.reference import ReferenceDataset
        ref = ReferenceDataset(data_path, coord_keys=("x", "t", "lognu"))
        self._axes_min_data = ref.axes_min
        self._axes_max_data = ref.axes_max

        # J grid: include x endpoints, exclude t endpoint, clip, then store float32.
        x_lin = np.linspace(self.xmin, self.xmax, self.j_nx, endpoint=True)
        t_lin = np.linspace(self.tmin, self.tmax, self.j_nt, endpoint=False)
        X, T = np.meshgrid(x_lin, t_lin, indexing="ij")
        xf = np.clip(X.reshape(-1), self._axes_min_data[0], self._axes_max_data[0])
        tf = np.clip(T.reshape(-1), self._axes_min_data[1], self._axes_max_data[1])
        self._j_xt = np.stack(
            [xf.astype(np.float32), tf.astype(np.float32)], axis=1
        )

    def load_reference(self, data_path: str):
        """The F10 reference stores the third axis as descending lognu; reorder it ascending."""
        from evaluation.reference import ReferenceDataset
        return ReferenceDataset(
            data_path, coord_keys=("x", "t", "lognu"), sort_axes=True
        )

    # Normalize queries by clipping to data bounds; no periodic mapping.
    def normalize_queries(self, Q: np.ndarray) -> np.ndarray:
        if self._axes_min_data is None:
            raise RuntimeError("F10: Call attach_data first")
        Q = np.asarray(Q, dtype=np.float64)
        if Q.ndim == 1:
            Q = Q.reshape(1, -1)
        return np.clip(Q, self._axes_min_data, self._axes_max_data)

    # Evaluation axes use data bounds with all endpoints included.
    def metric_axes_3d(self, reference, nx: int, nt: int, nmu: int):
        if self._axes_min_data is None:
            raise RuntimeError("F10: Call attach_data first")
        x = np.linspace(self._axes_min_data[0], self._axes_max_data[0], nx,
                        dtype=np.float32)
        t = np.linspace(self._axes_min_data[1], self._axes_max_data[1], nt,
                        dtype=np.float32)
        l = np.linspace(self._axes_min_data[2], self._axes_max_data[2], nmu,
                        dtype=np.float32)
        return x, t, l

    # Objective.
    @staticmethod
    def _F(u_val: float, j_val: float) -> float:
        """Shifted Rosenbrock-type objective evaluated in float64."""
        return float(100.0 * (u_val + 0.6 - (j_val + 0.6) ** 2) ** 2
                     + (j_val - 0.4) ** 2)

    def _dissipation_j(self, u_on_grid: np.ndarray, lognu: float) -> float:
        """J=nu*sum(u_x^2)*dx*dt; reshape the grid to (nx,nt) and difference along x."""
        U = np.asarray(u_on_grid, dtype=np.float64).reshape(self.j_nx, self.j_nt)
        ux = (U[2:, :] - U[:-2, :]) / (2.0 * self._j_dx)
        nu = float(np.exp(np.clip(lognu, self._axes_min_data[2],
                                  self._axes_max_data[2])))
        return float(nu * float(np.sum(ux * ux)) * self._j_dx * self._j_dt)

    def evaluate_fitness(self, decision: np.ndarray, state_provider) -> float:
        if self._j_xt is None:
            raise RuntimeError("F10: Call attach_data first; the J quadrature grid requires reference data")
        d = np.asarray(decision, dtype=float).reshape(-1)
        lognu = float(d[2])
        # Separate surrogate calls for the local point and the J quadrature grid.
        u_val = float(np.asarray(
            state_provider.evaluate(d.reshape(1, 3)), dtype=float
        ).reshape(-1)[0])
        j_val = self._j_via(d, lognu, state_provider)
        return self._F(u_val, j_val)

    def evaluate_fitness_for_pool(self, decision: np.ndarray, state_provider) -> float:
        """Pool objective arithmetic rounds J to float32 before evaluating F."""
        d = np.asarray(decision, dtype=float).reshape(-1)
        lognu = float(d[2])
        u_val = float(np.asarray(
            state_provider.evaluate(d.reshape(1, 3)), dtype=float
        ).reshape(-1)[0])
        j_val = float(np.float32(self._j_via(d, lognu, state_provider)))
        return self._F(u_val, j_val)

    def _j_via(self, d, lognu, state_provider) -> float:
        N = self._j_xt.shape[0]
        Qj = np.empty((N, 3), dtype=np.float64)
        Qj[:, 0:2] = self._j_xt.astype(np.float64)
        Qj[:, 2] = lognu
        uj = np.asarray(state_provider.evaluate(Qj), dtype=float).reshape(-1)
        return self._dissipation_j(uj, lognu)

    def violation(self, decision: np.ndarray, state_provider) -> float:
        # Unconstrained problem.
        return 0.0

    def constraint_components(self, decision, state_provider) -> list:
        return []   # Unconstrained: no constraint channels.

    def constraint_meta(self) -> list:
        return []   # Unconstrained: no constraint channels.