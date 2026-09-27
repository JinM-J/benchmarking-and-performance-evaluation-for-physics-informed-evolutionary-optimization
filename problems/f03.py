# problems/f03.py
"""F03 (paper F3): Allen-Cahn-type reaction-diffusion optimization.

    u_t - alpha*u_xx = -5*u^3+5*u, alpha=1e-4
    u(x,0) = -(2.15/pi)*x*cos(pi*x), x in [-5,5]
    u(-5,t) = u(5,t), t in [0,1]

Decision and query coordinates coincide: (x,t) in [-5,5] x [0,1].
The unconstrained Rastrigin-type objective uses z1=x-1.9004, z2=u+1.153:
    f = z1^2-10*cos(2*pi*z1)+10 + z2^2-10*cos(2*pi*z2)+10
Violation is identically zero; objective coordinates are not clamped."""
import numpy as np

from problems.base import (PDEProblem, PhysicsSpec, InitialCondition, BoundaryCondition,
                           LinearOperator, LinearOpTerm)
from problems.regions import face_region


class ReactionDiffusionPhysics(PhysicsSpec):
    """Reaction-diffusion PDE for F03."""

    def __init__(self, alpha: float, ic_c: float,
                 xmin: float, xmax: float, tmin: float, tmax: float):
        self.alpha = alpha
        self.ic_c = ic_c
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
        """Residual: u_t - alpha*u_xx - (-5*u^3+5*u)."""
        reaction = -5.0 * (u ** 3) + 5.0 * u
        return d[("t",)] - self.alpha * d[("x", "x")] - reaction

    def linear_operator(self):
        """Exact Allen-Cahn linear part: u_t - alpha*u_xx - 5*u=0.

        Retain the linear reaction +5*u; residual noise absorbs only -5*u^3."""
        return LinearOperator(terms=(
            LinearOpTerm((0, 1), 1.0),
            LinearOpTerm((2, 0), -self.alpha),
            LinearOpTerm((0, 0), -5.0),
        ))

    # ---------- IC: u(x,0) = -(ic_c/pi)*x*cos(pi*x) ----------
    def initial_conditions(self):
        return (
            InitialCondition(
                region=face_region(self._qb, axis=1, value=self.tmin,
                                   sample_axis=0),
                target=lambda Q: -(self.ic_c / np.pi) * Q[:, 0] * np.cos(np.pi * Q[:, 0]),
                derivative=(),
            ),
        )

    # Periodic state boundary: u(xmin,t)=u(xmax,t).
    def boundary_conditions(self):
        left = face_region(self._qb, axis=0, value=self.xmin, sample_axis=1)
        right = face_region(self._qb, axis=0, value=self.xmax, sample_axis=1)
        return (
            BoundaryCondition(kind="periodic", region_pair=(left, right)),
        )


class F03(PDEProblem):
    """Reaction-diffusion PDE with an unconstrained Rastrigin-type objective."""

    name = "F03"

    def __init__(
        self,
        alpha: float = 1e-4,
        ic_c: float = 2.15,
        xmin: float = -5.0,
        xmax: float = 5.0,
        tmin: float = 0.0,
        tmax: float = 1.0,
    ):
        self.alpha = alpha
        self.ic_c = ic_c
        self.xmin, self.xmax = xmin, xmax
        self.tmin, self.tmax = tmin, tmax
        self._physics = ReactionDiffusionPhysics(alpha, ic_c, xmin, xmax, tmin, tmax)

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
        return self.decision_bounds

    @property
    def physics(self) -> PhysicsSpec:
        return self._physics

    def decision_to_query(self, decision: np.ndarray) -> np.ndarray:
        return np.asarray(decision, dtype=float)

    # Unconstrained Rastrigin-type objective.
    def evaluate_fitness(self, decision: np.ndarray, state_provider) -> float:
        x = float(decision[0])
        t = float(decision[1])
        u = float(state_provider.evaluate(np.array([[x, t]]))[0])
        z1 = x - 1.9004
        z2 = u + 1.153
        return float(
            (z1 ** 2 - 10.0 * np.cos(2.0 * np.pi * z1) + 10.0)
            + (z2 ** 2 - 10.0 * np.cos(2.0 * np.pi * z2) + 10.0)
        )

    def violation(self, decision: np.ndarray, state_provider) -> float:
        # Unconstrained problem.
        return 0.0

    def constraint_components(self, decision, state_provider) -> list:
        return []   # Unconstrained: no constraint channels.

    def constraint_meta(self) -> list:
        return []   # Unconstrained: no constraint channels.