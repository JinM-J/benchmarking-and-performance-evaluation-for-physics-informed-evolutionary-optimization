# problems/f06.py
"""F06 (paper F6): local-signal optimization on a smooth background.

Linear convection-diffusion PDE with no parameter axis:
    u_t+c*u_x-nu*u_xx=s(x,t), c=0.5, nu=5e-4
    (x,t) in [-1,1] x [0,1]; decision and query spaces coincide.
Manufactured solution (stored-field interpolation still introduces error):
    u=b+g, b=0.3*exp(-t)*sin(pi*x)
    g=A*exp(-z^2), z=(x-x_s(t))/w, x_s=-0.25+0.5*t
    A=0.03, w=0.10
    s=-b+0.3*c*pi*exp(-t)*cos(pi*x)+nu*pi^2*b
      +(nu*A/w^2)*(2-4*z^2)*exp(-z^2)
Initial and boundary values follow the exact solution.

The unconstrained objective locates the right-side 5% packet response:
    q=(u-b(x,t))/A, eta=0.05
    F(x,t,u)=(1-q/eta)^2+max(0,-z)^2+alpha*(t-t_star)^2, alpha=0.2.
Full-field prediction error and this local objective measure different
properties; comparative method performance requires experiments.
On the exact field, q=exp(-z^2). All three terms vanish only at
t=0.7 and z=sqrt(log(20)), giving x=0.1+0.1*sqrt(log(20)) and F=0."""
import json

import numpy as np

from problems.base import (PDEProblem, PhysicsSpec, InitialCondition,
                           BoundaryCondition, LinearOperator, LinearOpTerm)
from problems.regions import face_region

# PDE and state constants must match dataset/generate_f06.py.
C_CONV = 0.5
NU = 5e-4
A_PKT = 0.03
W_PKT = 0.10
X_S0 = -0.25
T_STAR = 0.7
ALPHA_T = 0.2
RESPONSE_THRESHOLD = 0.05
X_STAR = X_S0 + C_CONV * T_STAR + W_PKT * np.sqrt(-np.log(RESPONSE_THRESHOLD))


def _background_np(x, t):
    return 0.3 * np.exp(-t) * np.sin(np.pi * x)


def _source_np(Q):
    """NumPy source s(Q) for PIGP and verification; Q has shape (N,2)=(x,t)."""
    x, t = Q[:, 0], Q[:, 1]
    b = _background_np(x, t)
    z = (x - (X_S0 + C_CONV * t)) / W_PKT
    return (-b
            + 0.3 * C_CONV * np.pi * np.exp(-t) * np.cos(np.pi * x)
            + NU * np.pi ** 2 * b
            + (NU * A_PKT / W_PKT ** 2) * (2 - 4 * z ** 2) * np.exp(-z ** 2))


class F06Physics(PhysicsSpec):
    """Linear forced convection-diffusion PDE with an exact PIGP operator."""

    def __init__(self, xmin, xmax, tmin, tmax):
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
        """Torch residual: u_t+c*u_x-nu*u_xx-s(x,t)."""
        import torch
        x, t = Q[:, 0:1], Q[:, 1:2]
        b = 0.3 * torch.exp(-t) * torch.sin(np.pi * x)
        z = (x - (X_S0 + C_CONV * t)) / W_PKT
        s = (-b
             + 0.3 * C_CONV * np.pi * torch.exp(-t) * torch.cos(np.pi * x)
             + NU * np.pi ** 2 * b
             + (NU * A_PKT / W_PKT ** 2) * (2 - 4 * z ** 2) * torch.exp(-z ** 2))
        return d[("t",)] + C_CONV * d[("x",)] - NU * d[("x", "x")] - s

    def linear_operator(self):
        """Complete linear operator: L[u]=u_t+c*u_x-nu*u_xx=s, with known forcing."""
        return LinearOperator(
            terms=(
                LinearOpTerm((0, 1), 1.0),
                LinearOpTerm((1, 0), C_CONV),
                LinearOpTerm((2, 0), -NU),
            ),
            rhs=_source_np,
        )

    def initial_conditions(self):
        return (
            InitialCondition(
                region=face_region(self._qb, axis=1, value=self.tmin,
                                   sample_axis=0),
                target=lambda Q: 0.3 * np.sin(np.pi * Q[:, 0])
                                 + A_PKT * np.exp(-((Q[:, 0] - X_S0)
                                                    / W_PKT) ** 2),
                derivative=(),
            ),
        )

    def boundary_conditions(self):
        """Exact Dirichlet values at both ends; the packet is small but nonzero there."""
        def _exact_at(xval):
            def _tgt(Q):
                t = Q[:, 1]
                return (_background_np(xval, t)
                        + A_PKT * np.exp(
                            -((xval - (X_S0 + C_CONV * t)) / W_PKT) ** 2))
            return _tgt
        return (
            BoundaryCondition(kind="dirichlet",
                              region=face_region(self._qb, axis=0,
                                                 value=self.xmin,
                                                 sample_axis=1),
                              target=_exact_at(self.xmin)),
            BoundaryCondition(kind="dirichlet",
                              region=face_region(self._qb, axis=0,
                                                 value=self.xmax,
                                                 sample_axis=1),
                              target=_exact_at(self.xmax)),
        )


class F06(PDEProblem):
    """Local-packet optimization problem for comparing field error and optimization utility."""

    name = "F06"

    def __init__(self, xmin=-1.0, xmax=1.0, tmin=0.0, tmax=1.0):
        self.xmin, self.xmax = xmin, xmax
        self.tmin, self.tmax = tmin, tmax
        self._physics = F06Physics(xmin, xmax, tmin, tmax)

        # Analytic objective constants; attach_data only checks dataset consistency.
        self.x_star = X_STAR
        self.t_star = T_STAR
        self.alpha_t = ALPHA_T
        self._ready = False

    # Decision and query spaces.
    @property
    def decision_bounds(self) -> np.ndarray:
        return np.array([[self.xmin, self.xmax],
                         [self.tmin, self.tmax]], dtype=np.float64)

    @property
    def query_bounds(self) -> np.ndarray:
        return self.decision_bounds

    @property
    def physics(self) -> PhysicsSpec:
        return self._physics

    def decision_to_query(self, decision: np.ndarray) -> np.ndarray:
        return np.asarray(decision, dtype=float)

    # Check that dataset generation constants match this definition.
    def attach_data(self, data_path: str) -> None:
        raw = np.load(data_path)
        meta = json.loads(str(raw["meta_json"]))
        pk = meta["packet"]
        for k, expect in (("A", A_PKT), ("w", W_PKT)):
            assert abs(float(pk[k]) - expect) < 1e-15, \
                f"Dataset packet {k}={pk[k]} does not match the problem value {expect}."
        assert abs(float(meta["nu"]) - NU) < 1e-15
        assert abs(float(meta["c"]) - C_CONV) < 1e-15
        self._ready = True

    # Reduced objective: state_provider supplies u; background b is analytic.
    def evaluate_fitness(self, decision: np.ndarray, state_provider) -> float:
        if not self._ready:
            raise RuntimeError("F06: Call attach_data first")
        x, t = float(decision[0]), float(decision[1])
        u = float(np.asarray(
            state_provider.evaluate(np.array([[x, t]])),
            dtype=float).reshape(-1)[0])
        q = (u - float(_background_np(x, t))) / A_PKT
        z = (x - X_S0 - C_CONV * t) / W_PKT
        return float((1 - q / RESPONSE_THRESHOLD) ** 2 + max(0.0, -z) ** 2
                     + self.alpha_t * (t - self.t_star) ** 2)

    # Unconstrained problem.
    def violation(self, decision, state_provider) -> float:
        return 0.0

    def constraint_components(self, decision, state_provider) -> list:
        return []

    def constraint_meta(self) -> list:
        return []
