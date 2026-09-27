# problems/f01.py
"""F01 (paper F1): one-dimensional Burgers-constrained optimization.

    u_t + u*u_x = nu*u_xx, nu=0.01/pi
    u(x,0) = -sin(pi*x), x in [-1,1]
    u(-1,t) = u(1,t) = 0, t in [0,1]

Decision and query coordinates are (x,t) in [-1,1] x [0,1].
The PDE determines u(x,t). In transformed coordinates:
    z1 = 3*(x+1), z2 = 3*(2-u)
    f = -sin(2*pi*z1)^3*sin(2*pi*z2)/(z1^3*(z1+z2))
    g1 = z1^2-z2+1 <= 0
    g2 = 1-z1+(z2-4)^2 <= 0
Only the objective clamps z1 below by eps to prevent division by zero;
the constraint transformation is not clamped."""
import numpy as np

from problems.base import (PDEProblem, PhysicsSpec, InitialCondition, BoundaryCondition,
                           LinearOperator, LinearOpTerm)
from problems.regions import face_region


class BurgersPhysics(PhysicsSpec):
    """Burgers PDE for F01."""

    def __init__(self, nu: float, xmin: float, xmax: float, tmin: float, tmax: float):
        self.nu = nu
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
        return {("t",), ("x",), ("x", "x")}

    def residual(self, Q, u, d):
        """Burgers residual: u_t + u*u_x - nu*u_xx."""
        return d[("t",)] + u * d[("x",)] - self.nu * d[("x", "x")]

    def linear_operator(self):
        """Exact linear part u_t - nu*u_xx=0; residual noise absorbs u*u_x."""
        return LinearOperator(terms=(
            LinearOpTerm((0, 1), 1.0),
            LinearOpTerm((2, 0), -self.nu),
        ))

    # ---------- IC: u(x,0) = -sin(pi*x) ----------
    def initial_conditions(self):
        return (
            InitialCondition(
                region=face_region(self._qb, axis=1, value=self.tmin,
                                   sample_axis=0),  # Sampling count is overridden by surrogate configuration.
                target=lambda Q: -np.sin(np.pi * Q[:, 0]),
                derivative=(),
            ),
        )

    # Two Dirichlet boundaries: u(-1,t)=u(1,t)=0.
    def boundary_conditions(self):
        return (
            BoundaryCondition(
                kind="dirichlet",
                region=face_region(self._qb, axis=0, value=self.xmin,
                                   sample_axis=1),
                target=lambda Q: np.zeros(Q.shape[0]),
            ),
            BoundaryCondition(
                kind="dirichlet",
                region=face_region(self._qb, axis=0, value=self.xmax,
                                   sample_axis=1),
                target=lambda Q: np.zeros(Q.shape[0]),
            ),
        )


class F01(PDEProblem):
    """Burgers problem with transformed objective and constraints."""

    name = "F01"

    def __init__(
        self,
        nu: float = 0.01 / np.pi,
        xmin: float = -1.0,
        xmax: float = 1.0,
        tmin: float = 0.0,
        tmax: float = 1.0,
    ):
        self.nu = nu
        self.xmin, self.xmax = xmin, xmax
        self.tmin, self.tmax = tmin, tmax
        self._physics = BurgersPhysics(nu, xmin, xmax, tmin, tmax)

    # Decision and query spaces.
    @property
    def decision_bounds(self) -> np.ndarray:
        return np.array(
            [[self.xmin, self.xmax],
             [self.tmin, self.tmax]],
            dtype=np.float64,
        )

    @property
    def query_bounds(self) -> np.ndarray:
        # Decision and query spaces coincide.
        return self.decision_bounds

    @property
    def physics(self) -> PhysicsSpec:
        return self._physics

    # Identity decision-to-query mapping.
    def decision_to_query(self, decision: np.ndarray) -> np.ndarray:
        return np.asarray(decision, dtype=float)

    # Transformed coordinates.
    def _transform(self, x, u):
        """Constraint transform: z1=3*(x+1), z2=3*(2-u), without clamping."""
        z1 = (x + 1) * 3
        z2 = (2 - u) * 3
        return z1, z2

    def _transform_clamped(self, x, u, eps=1e-10):
        """Clamp objective z1 below by eps to avoid division by z1^3=0."""
        z1 = (x + 1) * 3
        z2 = (2 - u) * 3
        if z1 <= eps:
            z1 = eps
        return z1, z2

    # Reduced objective: obtain u through state_provider.
    def evaluate_fitness(self, decision: np.ndarray, state_provider) -> float:
        x = float(decision[0])
        t = float(decision[1])
        u = float(state_provider.evaluate(np.array([[x, t]]))[0])
        z1, z2 = self._transform_clamped(x, u)
        return float(
            -(np.sin(2 * np.pi * z1) ** 3 * np.sin(2 * np.pi * z2))
            / (z1 ** 3 * (z1 + z2))
        )

    def violation(self, decision: np.ndarray, state_provider) -> float:
        x = float(decision[0])
        t = float(decision[1])
        u = float(state_provider.evaluate(np.array([[x, t]]))[0])
        z1, z2 = self._transform(x, u)
        g1 = z1 ** 2 - z2 + 1
        g2 = 1 - z1 + (z2 - 4) ** 2
        return float(g1) * (g1 > 0) + float(g2) * (g2 > 0)

    def constraint_components(self, decision, state_provider) -> list:
        """Separate g1/g2 inequality channels, consistent with violation()."""
        x = float(decision[0])
        t = float(decision[1])
        u = float(state_provider.evaluate(np.array([[x, t]]))[0])
        z1, z2 = self._transform(x, u)
        return [{"kind": "g", "value": float(z1 ** 2 - z2 + 1), "tol": 0.0},
                {"kind": "g", "value": float(1 - z1 + (z2 - 4) ** 2), "tol": 0.0}]

    def constraint_meta(self) -> list:
        """Constraint metadata matching constraint_components()."""
        return [{"kind": "g", "tol": 0.0}, {"kind": "g", "tol": 0.0}]
