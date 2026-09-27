# problems/f05.py
"""F05 (paper F5): constrained optimization with a periodic forced heat equation.

    u_t-alpha*u_xx = q(x,t), alpha=0.01
    q(x,t) = 2*sin(pi*x)*cos(2*pi*t)
    u(x,0) = 2.1*x*sin(pi*x), x in [0,3]
    u(0,t) = u(3,t), t in [0,2]

Decision and query coordinates are (x,t) in [0,3] x [0,2].
With z1=x, z2=u-0.5:
    f = -z1-z2
    g1 = -2*z1^4+8*z1^3-8*z1^2+z2-2 <= 0
    g2 = -4*z1^4+32*z1^3-88*z1^2+96*z1+z2-36 <= 0
Violation is the sum of the positive parts of g1 and g2."""
import numpy as np
import torch

from problems.base import (PDEProblem, PhysicsSpec, InitialCondition, BoundaryCondition,
                           LinearOperator, LinearOpTerm)
from problems.regions import face_region


class HeatSourcePeriodicPhysics(PhysicsSpec):
    """Periodic forced heat PDE for F05."""

    def __init__(self, alpha: float, xmin: float, xmax: float,
                 tmin: float, tmax: float):
        self.alpha = alpha
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
        return {("t",), ("x", "x")}

    def residual(self, Q, u, d):
        """Residual u_t-alpha*u_xx-q(x,t), with known forcing q."""
        q = 2.0 * torch.sin(np.pi * Q[:, 0:1]) * torch.cos(2.0 * np.pi * Q[:, 1:2])
        return d[("t",)] - self.alpha * d[("x", "x")] - q

    def linear_operator(self):
        """Complete linear operator: u_t-alpha*u_xx=q(x,t)."""
        return LinearOperator(
            terms=(
                LinearOpTerm((0, 1), 1.0),
                LinearOpTerm((2, 0), -self.alpha),
            ),
            rhs=lambda Q: 2.0 * np.sin(np.pi * Q[:, 0])
            * np.cos(2.0 * np.pi * Q[:, 1]),
        )

    # ---------- IC: u(x,0) = 2.1*x*sin(pi*x) ----------
    def initial_conditions(self):
        return (
            InitialCondition(
                region=face_region(self._qb, axis=1, value=self.tmin,
                                   sample_axis=0),
                target=lambda Q: 2.1 * Q[:, 0] * np.sin(np.pi * Q[:, 0]),
                derivative=(),
            ),
        )

    # Periodic state boundary: u(0,t)=u(3,t).
    def boundary_conditions(self):
        left = face_region(self._qb, axis=0, value=self.xmin, sample_axis=1)
        right = face_region(self._qb, axis=0, value=self.xmax, sample_axis=1)
        return (
            BoundaryCondition(kind="periodic", region_pair=(left, right)),
        )


class F05(PDEProblem):
    """Forced heat PDE with a linear objective and two polynomial constraints."""

    name = "F05"

    def __init__(
        self,
        alpha: float = 0.01,
        xmin: float = 0.0,
        xmax: float = 3.0,
        tmin: float = 0.0,
        tmax: float = 2.0,
    ):
        self.alpha = alpha
        self.xmin, self.xmax = xmin, xmax
        self.tmin, self.tmax = tmin, tmax
        self._physics = HeatSourcePeriodicPhysics(alpha, xmin, xmax, tmin, tmax)

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

    def evaluate_fitness(self, decision: np.ndarray, state_provider) -> float:
        x = float(decision[0])
        t = float(decision[1])
        u = float(state_provider.evaluate(np.array([[x, t]]))[0])
        z1 = x
        z2 = u - 0.5
        return float(-z1 - z2)

    def violation(self, decision: np.ndarray, state_provider) -> float:
        x = float(decision[0])
        t = float(decision[1])
        u = float(state_provider.evaluate(np.array([[x, t]]))[0])
        z1 = x
        z2 = u - 0.5
        g1 = -2.0 * z1 ** 4 + 8.0 * z1 ** 3 - 8.0 * z1 ** 2 + z2 - 2.0
        g2 = -4.0 * z1 ** 4 + 32.0 * z1 ** 3 - 88.0 * z1 ** 2 + 96.0 * z1 + z2 - 36.0
        # Clamp each violation below by zero before summing.
        return (float(g1) if g1 > 0 else 0.0) + (float(g2) if g2 > 0 else 0.0)

    def constraint_components(self, decision, state_provider) -> list:
        """Separate g1/g2 inequality channels, consistent with violation()."""
        x = float(decision[0])
        t = float(decision[1])
        u = float(state_provider.evaluate(np.array([[x, t]]))[0])
        z1, z2 = x, u - 0.5
        g1 = -2.0 * z1 ** 4 + 8.0 * z1 ** 3 - 8.0 * z1 ** 2 + z2 - 2.0
        g2 = -4.0 * z1 ** 4 + 32.0 * z1 ** 3 - 88.0 * z1 ** 2 + 96.0 * z1 + z2 - 36.0
        return [{"kind": "g", "value": float(g1), "tol": 0.0},
                {"kind": "g", "value": float(g2), "tol": 0.0}]

    def constraint_meta(self) -> list:
        """Constraint metadata matching constraint_components()."""
        return [{"kind": "g", "tol": 0.0}, {"kind": "g", "tol": 0.0}]
