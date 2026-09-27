# problems/f04.py
"""F04 (paper F4): forced heat-equation optimization.

    u_t-u_xx = beta_u*(source(x,t)-u), beta_u=1.5
    source(x,t) = b(x) dot u_vec(t)
    b(x) is a four-segment one-hot basis on [0,pi].
    u_vec_i(t) = 1.1+5*sin(t/4+i/10), i=1..4
    u(x,0)=0; u(0,t)=u(pi,t)=0

Decision and query coordinates are (x,t) in [0,pi] x [0,2].
The unconstrained six-hump camel objective uses z1=x, z2=u-1:
    f = 4*z2^2-2.1*z2^4+z2^6/3+z1*z2-4*z1^2+4*z1^4
The known source does not depend on model parameters, so detaching it
in residual computation preserves parameter gradients."""
import numpy as np
import torch

from problems.base import (PDEProblem, PhysicsSpec, InitialCondition, BoundaryCondition,
                           LinearOperator, LinearOpTerm)
from problems.regions import face_region


class HeatSourcePhysics(PhysicsSpec):
    """Forced heat PDE for F04."""

    def __init__(self, beta_u: float, xmin: float, xmax: float,
                 tmin: float, tmax: float):
        self.beta_u = beta_u
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

    def _source(self, x, t):
        """source(x,t)=sum_i b_i(x)*u_vec_i(t), using four one-hot segments."""
        x1 = x.reshape(-1)
        b_mat = torch.zeros((x1.shape[0], 4), device=x.device)
        pi = float(np.pi)
        b_mat[:, 0] = ((x1 >= 0.0) & (x1 < pi / 4.0)).float()
        b_mat[:, 1] = ((x1 >= pi / 4.0) & (x1 < pi / 2.0)).float()
        b_mat[:, 2] = ((x1 >= pi / 2.0) & (x1 < 3.0 * pi / 4.0)).float()
        b_mat[:, 3] = ((x1 >= 3.0 * pi / 4.0) & (x1 <= pi)).float()

        t_col = t.reshape(-1, 1).repeat(1, 4)
        i_vals = torch.arange(1, 5, device=t.device).float().unsqueeze(0)
        u_vec = 1.1 + 5.0 * torch.sin(t_col / 4.0 + i_vals / 10.0)

        return torch.sum(b_mat * u_vec, dim=1, keepdim=True)

    def residual(self, Q, u, d):
        """Residual u_t-u_xx-beta_u*(source-u), with detached known forcing."""
        source = self._source(Q[:, 0].detach(), Q[:, 1].detach())
        return d[("t",)] - d[("x", "x")] - self.beta_u * (source - u)

    def _source_np(self, Q):
        """NumPy source for the PIGP right-hand side, using the same formula as _source."""
        x, t = Q[:, 0], Q[:, 1]
        pi = float(np.pi)
        b = np.zeros((x.shape[0], 4))
        b[:, 0] = (x >= 0.0) & (x < pi / 4.0)
        b[:, 1] = (x >= pi / 4.0) & (x < pi / 2.0)
        b[:, 2] = (x >= pi / 2.0) & (x < 3.0 * pi / 4.0)
        b[:, 3] = (x >= 3.0 * pi / 4.0) & (x <= pi)
        i_vals = np.arange(1, 5, dtype=float)
        u_vec = 1.1 + 5.0 * np.sin(t[:, None] / 4.0 + i_vals[None, :] / 10.0)
        return (b * u_vec).sum(axis=1)

    def linear_operator(self):
        """Complete linear operator: u_t-u_xx+beta*u=beta*source(x,t)."""
        return LinearOperator(
            terms=(
                LinearOpTerm((0, 1), 1.0),
                LinearOpTerm((2, 0), -1.0),
                LinearOpTerm((0, 0), self.beta_u),
            ),
            rhs=lambda Q: self.beta_u * self._source_np(Q),
        )

    # ---------- IC: u(x,0) = 0 ----------
    def initial_conditions(self):
        return (
            InitialCondition(
                region=face_region(self._qb, axis=1, value=self.tmin,
                                   sample_axis=0),
                target=lambda Q: np.zeros(Q.shape[0]),
                derivative=(),
            ),
        )

    # ---------- BC: u(0,t) = u(pi,t) = 0 ----------
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


class F04(PDEProblem):
    """Forced heat PDE with an unconstrained six-hump objective."""

    name = "F04"

    def __init__(
        self,
        beta_u: float = 1.5,
        xmin: float = 0.0,
        xmax: float = float(np.pi),
        tmin: float = 0.0,
        tmax: float = 2.0,
    ):
        self.beta_u = beta_u
        self.xmin, self.xmax = xmin, xmax
        self.tmin, self.tmax = tmin, tmax
        self._physics = HeatSourcePhysics(beta_u, xmin, xmax, tmin, tmax)

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

    # Unconstrained objective.
    def evaluate_fitness(self, decision: np.ndarray, state_provider) -> float:
        x = float(decision[0])
        t = float(decision[1])
        u = float(state_provider.evaluate(np.array([[x, t]]))[0])
        z1 = x
        z2 = u - 1.0
        return float(
            4.0 * z2 ** 2 - 2.1 * z2 ** 4 + (1.0 / 3.0) * z2 ** 6
            + z1 * z2 - 4.0 * z1 ** 2 + 4.0 * z1 ** 4
        )

    def violation(self, decision: np.ndarray, state_provider) -> float:
        # Unconstrained problem.
        return 0.0

    def constraint_components(self, decision, state_provider) -> list:
        return []   # Unconstrained: no constraint channels.

    def constraint_meta(self) -> list:
        return []   # Unconstrained: no constraint channels.
