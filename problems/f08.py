# problems/f08.py
"""F08 (paper F8): Kuramoto-Sivashinsky-constrained optimization.

    u_t+alpha*u*u_x+beta*u_xx+gamma*u_xxxx=0
    alpha=100/16, beta=100/16^2, gamma=100/16^4
    u(x,0)=cos(x)*(1+sin(x)), x in [0,2]
    u(0,t)=u(2,t), u_x(0,t)=u_x(2,t), t in [0,1]

Decision and query coordinates coincide: (x,t) in [0,2] x [0,1].
    f = x^2+(u-1)^2
    abs(u-x^2) <= eps_eq (1e-3)
The equality tolerance band is represented by inequalities
    g1=(u-x^2)-eps_eq<=0; g2=-(u-x^2)-eps_eq<=0.
Violation sums their positive parts. The residual requires fourth-order
spatial derivatives."""
import numpy as np
from scipy.interpolate import RegularGridInterpolator

from evaluation.reference import ReferenceDataset

from problems.base import (PDEProblem, PhysicsSpec, InitialCondition, BoundaryCondition,
                           LinearOperator, LinearOpTerm, Region)
from problems.regions import face_region


class KuramotoSivashinskyPhysics(PhysicsSpec):
    """Kuramoto-Sivashinsky PDE for F08."""

    def __init__(self, alpha: float, beta: float, gamma: float,
                 xmin: float, xmax: float, tmin: float, tmax: float):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.xmin, self.xmax = xmin, xmax
        self.tmin, self.tmax = tmin, tmax
        self._qb = np.array([[xmin, xmax], [tmin, tmax]], dtype=float)

    @property
    def coordinate_names(self) -> tuple:
        return ("x", "t")

    @property
    def time_axis(self):
        return 1

    def required_derivatives(self) -> set:
        return {("t",), ("x",), ("x", "x"), ("x", "x", "x", "x")}

    def residual(self, Q, u, d):
        """KS residual: u_t+alpha*u*u_x+beta*u_xx+gamma*u_xxxx."""
        return (d[("t",)] + self.alpha * u * d[("x",)]
                + self.beta * d[("x", "x")] + self.gamma * d[("x", "x", "x", "x")])

    def linear_operator(self):
        """Exact KS linear part: u_t+beta*u_xx+gamma*u_xxxx=0.

        Residual noise absorbs alpha*u*u_x; Hermite recurrence provides
        the closed-form fourth derivatives for operator kernels."""
        return LinearOperator(terms=(
            LinearOpTerm((0, 1), 1.0),
            LinearOpTerm((2, 0), self.beta),
            LinearOpTerm((4, 0), self.gamma),
        ))

    # ---------- IC: u(x,0) = cos(x)*(1+sin(x)) ----------
    def initial_conditions(self):
        return (
            InitialCondition(
                region=face_region(self._qb, axis=1, value=self.tmin,
                                   sample_axis=0),
                target=lambda Q: np.cos(Q[:, 0]) * (1.0 + np.sin(Q[:, 0])),
                derivative=(),
            ),
        )

    # Periodic state and derivative: u(xmin)=u(xmax), u_x(xmin)=u_x(xmax).
    def boundary_conditions(self):
        left = face_region(self._qb, axis=0, value=self.xmin, sample_axis=1)
        right = face_region(self._qb, axis=0, value=self.xmax, sample_axis=1)
        return (
            BoundaryCondition(kind="periodic_derivative",
                              region_pair=(left, right),
                              derivative=("x",)),
        )


class PeriodicKSPhysics(KuramotoSivashinskyPhysics):

    def initial_conditions(self):

        def sample(n):
            x = np.linspace(self.xmin, self.xmax, n, endpoint=False)
            return np.column_stack((x, np.full(n, self.tmin)))
        region = Region(contains_fn=lambda q: np.isclose(q[:, 1], self.tmin) & (q[:, 0] >= self.xmin) & (q[:, 0] < self.xmax), sample_fn=sample)
        return (InitialCondition(region=region, target=lambda q: np.cos(q[:, 0]) * (1 + np.sin(q[:, 0]))),)

    def boundary_conditions(self):

        def face(x_value):

            def sample(n):
                t = np.linspace(self.tmin, self.tmax, n + 1)[1:]
                return np.column_stack((np.full(n, x_value), t))
            return Region(contains_fn=lambda q: np.isclose(q[:, 0], x_value) & (q[:, 1] > self.tmin) & (q[:, 1] <= self.tmax), sample_fn=sample)
        pair = (face(self.xmin), face(self.xmax))
        return tuple((BoundaryCondition(kind='periodic_derivative', region_pair=pair, derivative=('x',) * order, include_value=order == 1) for order in (1, 2, 3)))

class PeriodicKSReference(ReferenceDataset):

    def __init__(self, path, *, xmin, xmax, tmin):
        super().__init__(path)
        if self.u_grid.shape != (len(self.x_grid), len(self.t_grid)):
            raise ValueError('KS reference axes do not match the stored state')
        if self.x_grid[0] != xmin or self.x_grid[-1] >= xmax:
            raise ValueError('Expected a half-open periodic KS spatial grid')
        self._xmin, self._xmax, self._tmin = (xmin, xmax, tmin)
        self.interpolator = RegularGridInterpolator((np.append(self.x_grid, xmax), self.t_grid), np.concatenate((self.u_grid, self.u_grid[:1]), axis=0))

    def query(self, Q):
        q = np.asarray(Q, dtype=float).reshape(-1, 2).copy()
        q[:, 0] = self._xmin + np.mod(q[:, 0] - self._xmin, self._xmax - self._xmin)
        q[:, 1] = np.clip(q[:, 1], self.t_min, self.t_max)
        values = self.interpolator(q)
        initial = q[:, 1] == self._tmin
        x = q[initial, 0]
        values[initial] = np.cos(x) * (1 + np.sin(x))
        return values


class F08(PDEProblem):
    """KS PDE with a quadratic objective and an equality tolerance band."""

    name = "F08"

    def __init__(
        self,
        alpha: float = 100.0 / 16.0,
        beta: float = 100.0 / (16.0 ** 2),
        gamma: float = 100.0 / (16.0 ** 4),
        eps_eq: float = 1e-3,
        xmin: float = 0.0,
        xmax: float = 2.0,
        tmin: float = 0.0,
        tmax: float = 1.0,
    ):
        self.alpha, self.beta, self.gamma = alpha, beta, gamma
        self.eps_eq = eps_eq
        self.xmin, self.xmax = xmin, xmax
        self.tmin, self.tmax = tmin, tmax
        self._physics = PeriodicKSPhysics(
            alpha, beta, gamma, xmin, xmax, tmin, tmax
        )

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

    def load_reference(self, data_path):
        return PeriodicKSReference(data_path, xmin=self.xmin, xmax=self.xmax, tmin=self.tmin)

    def evaluate_fitness(self, decision: np.ndarray, state_provider) -> float:
        x = float(decision[0])
        t = float(decision[1])
        u = float(state_provider.evaluate(np.array([[x, t]]))[0])
        return float(x ** 2 + (u - 1.0) ** 2)

    def violation(self, decision: np.ndarray, state_provider) -> float:
        x = float(decision[0])
        t = float(decision[1])
        u = float(state_provider.evaluate(np.array([[x, t]]))[0])
        h = u - x ** 2
        g1 = h - self.eps_eq
        g2 = (-h) - self.eps_eq
        return float(g1) * (g1 > 0) + float(g2) * (g2 > 0)

    def constraint_components(self, decision, state_provider) -> list:
        """Single equality channel h=u-x^2 with problem tolerance eps_eq."""
        x = float(decision[0])
        t = float(decision[1])
        u = float(state_provider.evaluate(np.array([[x, t]]))[0])
        return [{"kind": "h", "value": float(u - x ** 2), "tol": self.eps_eq}]

    def constraint_meta(self) -> list:
        """Constraint metadata matching constraint_components()."""
        return [{"kind": "h", "tol": self.eps_eq}]
