# problems/f02.py
"""F02 (paper F2): wave-equation-constrained optimization.

    u_tt - c^2*u_xx = 0, c=2.0
    u(x,0) = -2.5*sin(pi*x)+2.6*sin(4*pi*x), x in [0,5]
    u_t(x,0) = 0
    u(0,t) = u(5,t) = 0, t in [0,1]

Decisions (x,t,z) lie in [0,5] x [0,1] x [0,10]. The surrogate queries
only (x,t); z enters the objective and constraints but not the PDE.
With z1=2*x, z2=u, z3=z:
    f = 1000-z1^2-2*z2^2-z3^2-z1*z2-z1*z3
    abs(z1^2+z2^2+z3^2-25) <= eps1 (0.1)
    abs(8*z1+14*z2+7*z3-56) <= eps2 (0.1)
Violation sums the positive parts of deviations exceeding these tolerances.
The residual needs second temporal derivatives and zero initial velocity."""
import numpy as np

from problems.base import (PDEProblem, PhysicsSpec, InitialCondition, BoundaryCondition,
                           LinearOperator, LinearOpTerm)
from problems.regions import face_region


class WavePhysics(PhysicsSpec):
    """Wave PDE for F02."""

    def __init__(self, c: float, xmin: float, xmax: float, tmin: float, tmax: float):
        self.c = c
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
        # The wave equation requires second derivatives in time and space.
        return {("t", "t"), ("x", "x")}

    def residual(self, Q, u, d):
        """Wave residual: u_tt - c^2*u_xx."""
        return d[("t", "t")] - (self.c ** 2) * d[("x", "x")]

    def linear_operator(self):
        """Complete linear wave operator: u_tt - c^2*u_xx=0."""
        return LinearOperator(terms=(
            LinearOpTerm((0, 2), 1.0),
            LinearOpTerm((2, 0), -(self.c ** 2)),
        ))

    # Initial displacement and zero initial velocity.
    def initial_conditions(self):
        t0 = face_region(self._qb, axis=1, value=self.tmin, sample_axis=0)
        return (
            InitialCondition(
                region=t0,
                target=lambda Q: -2.5 * np.sin(np.pi * Q[:, 0])
                                 + 2.6 * np.sin(4.0 * np.pi * Q[:, 0]),
                derivative=(),
            ),
            InitialCondition(
                region=t0,
                target=lambda Q: np.zeros(Q.shape[0]),
                derivative=("t",),   # Initial velocity u_t(x,0)=0.
            ),
        )

    # ---------- BC: u(0,t) = u(5,t) = 0 ----------
    def boundary_conditions(self):
        return (
            BoundaryCondition(
                kind="dirichlet",
                region=face_region(self._qb, axis=0, value=self.xmin, sample_axis=1),
                target=lambda Q: np.zeros(Q.shape[0]),
            ),
            BoundaryCondition(
                kind="dirichlet",
                region=face_region(self._qb, axis=0, value=self.xmax, sample_axis=1),
                target=lambda Q: np.zeros(Q.shape[0]),
            ),
        )


class F02(PDEProblem):
    """Wave PDE with three decisions (x,t,z) and two equality tolerance bands."""

    name = "F02"

    def __init__(
        self,
        c: float = 2.0,
        xmin: float = 0.0,
        xmax: float = 5.0,
        tmin: float = 0.0,
        tmax: float = 1.0,
        zmin: float = 0.0,
        zmax: float = 10.0,
        eps1: float = 0.1,
        eps2: float = 0.1,
    ):
        self.c = c
        self.xmin, self.xmax = xmin, xmax
        self.tmin, self.tmax = tmin, tmax
        self.zmin, self.zmax = zmin, zmax
        self.eps1, self.eps2 = eps1, eps2
        self._physics = WavePhysics(c, xmin, xmax, tmin, tmax)

    # Three decision dimensions and two query dimensions.
    @property
    def decision_bounds(self) -> np.ndarray:
        return np.array(
            [[self.xmin, self.xmax],
             [self.tmin, self.tmax],
             [self.zmin, self.zmax]],
            dtype=np.float64,
        )

    @property
    def query_bounds(self) -> np.ndarray:
        return np.array(
            [[self.xmin, self.xmax],
             [self.tmin, self.tmax]],
            dtype=np.float64,
        )

    @property
    def physics(self) -> PhysicsSpec:
        return self._physics

    def decision_to_query(self, decision: np.ndarray) -> np.ndarray:
        """Discard z from surrogate queries; retain (x,t)."""
        return np.asarray(decision, dtype=float)[:2]

    # Objective and constraints.
    def evaluate_fitness(self, decision: np.ndarray, state_provider) -> float:
        x = float(decision[0])
        t = float(decision[1])
        z = float(decision[2])
        u = float(state_provider.evaluate(np.array([[x, t]]))[0])
        z1 = 2.0 * x
        z2 = u
        z3 = z
        return float(1000.0 - z1 ** 2 - 2.0 * z2 ** 2 - z3 ** 2 - z1 * z2 - z1 * z3)

    def violation(self, decision: np.ndarray, state_provider) -> float:
        x = float(decision[0])
        t = float(decision[1])
        z = float(decision[2])
        u = float(state_provider.evaluate(np.array([[x, t]]))[0])
        z1 = 2.0 * x
        z2 = u
        z3 = z
        dev1 = abs(z1 ** 2 + z2 ** 2 + z3 ** 2 - 25.0)
        dev2 = abs(8.0 * z1 + 14.0 * z2 + 7.0 * z3 - 56.0)
        return float(max(0.0, dev1 - self.eps1) + max(0.0, dev2 - self.eps2))

    def constraint_components(self, decision, state_provider) -> list:
        """Two equality channels with the problem tolerances eps1 and eps2."""
        x = float(decision[0])
        t = float(decision[1])
        z = float(decision[2])
        u = float(state_provider.evaluate(np.array([[x, t]]))[0])
        z1, z2, z3 = 2.0 * x, u, z
        return [{"kind": "h", "value": float(z1 ** 2 + z2 ** 2 + z3 ** 2 - 25.0),
                 "tol": self.eps1},
                {"kind": "h", "value": float(8.0 * z1 + 14.0 * z2 + 7.0 * z3 - 56.0),
                 "tol": self.eps2}]

    def constraint_meta(self) -> list:
        """Metadata for the two equality channels and their tolerances."""
        return [{"kind": "h", "tol": self.eps1}, {"kind": "h", "tol": self.eps2}]
