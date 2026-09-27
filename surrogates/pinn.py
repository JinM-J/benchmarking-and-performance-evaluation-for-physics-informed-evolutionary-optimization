# surrogates/pinn.py
"""Physics-informed neural network on generic query coordinates.

problem.physics defines PDE residuals and IC/BC locations, values and derivative
requirements. This surrogate implements autograd, losses, training and the
learning-rate schedule. Derivatives follow axis sequences returned by
physics.required_derivatives(), including mixed and high-order derivatives.
Convert NumPy IC/BC targets to torch.float32. GPU training is not guaranteed
bitwise reproducible; validation uses structural criteria.
"""
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import grad

from surrogates.base import Surrogate


class PINNNet(nn.Module):
    """SiLU MLP with forward(*coords), e.g. forward(x,t) or forward(x,t,mu)."""

    def __init__(self, layers):
        super().__init__()
        self.net = nn.Sequential()
        for i in range(len(layers) - 1):
            self.net.add_module(f"layer_{i}", nn.Linear(layers[i], layers[i + 1]))
            if i < len(layers) - 2:
                self.net.add_module(f"act_{i}", nn.SiLU())

    def forward(self, *coords):
        inputs = torch.cat(list(coords), dim=1)
        return self.net(inputs)


class PINNSurrogate(Surrogate):
    name = "PINN"

    def __init__(self, config: dict = None):
        config = config or {}
        self.layers = config.get("layers", [2, 80, 80, 80, 80, 1])
        self.ever_epochs = int(config.get("ever_epochs", 300))
        self.ic_points = int(config.get("ic_points", 500))
        self.bc_points = int(config.get("bc_points", 500))
        # Additional BC collocation budget, e.g. F07 hole walls use obs_bc_points.
        self.obs_bc_points = int(config.get("obs_bc_points", self.bc_points))
        self.res_points = int(config.get("res_points", 10000))
        self.lr_max = float(config.get("lr_max", 0.008))
        self.lr_min = float(config.get("lr_min", 0.000001))
        self.lr = float(config.get("lr", 0.008))
        self.lr_schedule = config.get("lr_schedule", "exponential")  # exponential | none
        self.data_weight = float(config.get("data_weight", 5.0))
        self.res_weight = float(config.get("res_weight", 1.0))
        self.ic_weight = float(config.get("ic_weight", 1.0))
        self.bc_weight = float(config.get("bc_weight", 1.0))
        self.ic_t_weight = float(config.get("ic_t_weight", 1.0))
        # Optional nondimensionalization; defaults preserve protocol behavior.
        # Output remains physical; autograd supplies chain factors for input/output transforms.
        self.normalize_inputs = bool(config.get("normalize_inputs", False))
        self.output_offset = float(config.get("output_offset", 0.0))
        self.output_scale = float(config.get("output_scale", 1.0))
        self.residual_scale = float(config.get("residual_scale", 1.0))
        self.seed_model_init = bool(config.get("seed_model_init", False))
        axes = config.get("input_axes", None)
        self.input_axes = None if axes is None else tuple(int(a) for a in axes)

        self.model = None
        self.optimizer = None
        self.problem = None

    # ------------------------------------------------------------------
    # Build IC/BC collocation from the physics interface.
    # ------------------------------------------------------------------
    def setup(self, problem, seed: int, config: dict = None):
        if config:
            self.layers = config.get("layers", self.layers)
            self.ever_epochs = int(config.get("ever_epochs", self.ever_epochs))
            self.ic_points = int(config.get("ic_points", self.ic_points))
            self.bc_points = int(config.get("bc_points", self.bc_points))
            self.obs_bc_points = int(config.get("obs_bc_points", self.obs_bc_points))
            self.res_points = int(config.get("res_points", self.res_points))
            self.lr_max = float(config.get("lr_max", self.lr_max))
            self.lr_min = float(config.get("lr_min", self.lr_min))
            self.lr = float(config.get("lr", self.lr))
            self.lr_schedule = config.get("lr_schedule", self.lr_schedule)
            self.data_weight = float(config.get("data_weight", self.data_weight))
            self.res_weight = float(config.get("res_weight", self.res_weight))
            self.ic_weight = float(config.get("ic_weight", self.ic_weight))
            self.bc_weight = float(config.get("bc_weight", self.bc_weight))
            self.ic_t_weight = float(config.get("ic_t_weight", self.ic_t_weight))
            self.normalize_inputs = bool(config.get("normalize_inputs", self.normalize_inputs))
            self.output_offset = float(config.get("output_offset", self.output_offset))
            self.output_scale = float(config.get("output_scale", self.output_scale))
            self.residual_scale = float(config.get("residual_scale", self.residual_scale))
            self.seed_model_init = bool(config.get("seed_model_init", self.seed_model_init))
            axes = config.get("input_axes", self.input_axes)
            self.input_axes = None if axes is None else tuple(int(a) for a in axes)

        if self.output_scale <= 0.0 or self.residual_scale <= 0.0:
            raise ValueError("PINN output_scale and residual_scale must be positive")

        self.problem = problem
        self.seed = seed
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        qb = np.asarray(problem.query_bounds, dtype=float)
        self._dim = qb.shape[0]
        if self.input_axes is not None:
            if (not self.input_axes
                    or len(set(self.input_axes)) != len(self.input_axes)
                    or min(self.input_axes) < 0 or max(self.input_axes) >= self._dim):
                raise ValueError(
                    f"PINN input_axes={self.input_axes} is invalid for a {self._dim}-dimensional query space"
                )
            # PI-DeepONet replaces this network after setup and controls axis reduction itself.
            # For ordinary PINN, layers[0] must match the selected physical axes.
            if type(self) is PINNSurrogate and self.layers[0] != len(self.input_axes):
                raise ValueError(
                    f"PINN layers[0]={self.layers[0]} != "
                    f"len(input_axes)={len(self.input_axes)}"
                )
        self._qmin = torch.tensor(qb[:, 0], dtype=torch.float32, device=self.device)
        self._qspan = torch.tensor(qb[:, 1] - qb[:, 0], dtype=torch.float32, device=self.device)
        if self.seed_model_init:
            # Seed network initialization only when explicitly enabled by the protocol.
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        # Initial-condition collocation.
        # Keep targets as (n,1) columns to match network output for pointwise MSE.
        # Shape (n,) would broadcast to pairwise (n,n) differences.
        self.ic_terms = []
        for ic in problem.physics.initial_conditions():
            n = ic.n_points if ic.n_points is not None else self.ic_points
            Q_np = ic.region.sample(n)                                  # (n, dim)
            target_np = np.asarray(ic.target(Q_np), dtype=float).reshape(-1, 1)
            self.ic_terms.append({
                "Q": torch.tensor(Q_np, dtype=torch.float32, device=self.device),
                "target": torch.tensor(target_np, dtype=torch.float32, device=self.device),
                "deriv": tuple(ic.derivative),
            })

        # Boundary-condition collocation.
        # Use each BC n_points when supplied; otherwise resolve its points_key
        # against the surrogate collocation budgets from the protocol.
        bc_budgets = {"bc_points": self.bc_points,
                      "obs_bc_points": self.obs_bc_points}
        self.bc_terms = []
        for bc in problem.physics.boundary_conditions():
            if bc.kind == "dirichlet":
                n = bc.n_points if bc.n_points is not None else bc_budgets[bc.points_key]
                Q_np = bc.region.sample(n)
                target_np = np.asarray(bc.target(Q_np), dtype=float).reshape(-1, 1)
                self.bc_terms.append({
                    "kind": "dirichlet",
                    "Q": torch.tensor(Q_np, dtype=torch.float32, device=self.device),
                    "target": torch.tensor(target_np, dtype=torch.float32, device=self.device),
                })
            elif bc.kind == "neumann":
                # Derivative BCs, including symmetry and prescribed flux, compare
                # the bc.derivative directional derivative with the possibly parameter-dependent target.
                n = bc.n_points if bc.n_points is not None else bc_budgets[bc.points_key]
                Q_np = bc.region.sample(n)
                target_np = np.asarray(bc.target(Q_np), dtype=float).reshape(-1, 1)
                self.bc_terms.append({
                    "kind": "neumann",
                    "Q": torch.tensor(Q_np, dtype=torch.float32, device=self.device),
                    "target": torch.tensor(target_np, dtype=torch.float32, device=self.device),
                    "deriv": tuple(bc.derivative),
                })
            elif bc.kind in ("periodic", "periodic_derivative"):
                ra, rb = bc.region_pair
                n = bc.n_points if bc.n_points is not None else bc_budgets[bc.points_key]
                self.bc_terms.append({
                    "kind": bc.kind,
                    "Qa": torch.tensor(ra.sample(n), dtype=torch.float32, device=self.device),
                    "Qb": torch.tensor(rb.sample(n), dtype=torch.float32, device=self.device),
                    "deriv": tuple(bc.derivative),
                })
            else:
                raise KeyError(f"Unknown boundary type: {bc.kind}")

        # Interior residual collocation.
        # Prefer problem-specific sampling, such as rejection in the F07 perforated domain.
        # Otherwise sample each coordinate uniformly.
        Q_res_np = problem.sample_interior_queries(self.res_points, seed)
        if Q_res_np is not None:
            self.Q_res = torch.tensor(np.asarray(Q_res_np, dtype=float),
                                      dtype=torch.float32, device=self.device)
        else:
            g = torch.Generator(device=self.device)
            g.manual_seed(seed)
            cols = []
            for j in range(self._dim):
                c = self._qmin[j] + self._qspan[j] * torch.rand(
                    self.res_points, 1, generator=g, device=self.device)
                cols.append(c)
            self.Q_res = torch.cat(cols, dim=1)   # (res_points, dim)

        self.model = PINNNet(self.layers).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.last_loss_components = None

    def _set_lr_by_gen(self, gen: int, maxgen: int) -> float:
        g = float(np.clip(gen, 0, maxgen))
        lr_now = self.lr_max * ((self.lr_min / self.lr_max) ** (g / maxgen))
        for pg in self.optimizer.param_groups:
            pg["lr"] = float(lr_now)
        return float(lr_now)

    # ------------------------------------------------------------------
    # Axis-sequence autograd supports mixed and high-order derivatives.
    # ------------------------------------------------------------------
    def _derivatives(self, u, Q, required):
        """Compute required derivatives, e.g. {("x","x"),("t",)}.

        Return {axes_tuple:tensor}; physics.coordinate_names defines axis_index.
        """
        axis_index = {name: i for i, name in
                      enumerate(self.problem.physics.coordinate_names)}
        d = {}
        for axes in required:
            out = u
            for ax in axes:
                gfull = grad(out, Q, grad_outputs=torch.ones_like(out),
                             create_graph=True)[0]
                out = gfull[:, axis_index[ax]].unsqueeze(-1)
            d[axes] = out
        return d

    def _forward_physical(self, model, Q):
        """Map physical coordinates Q to physical state u, with optional internal scaling."""
        X = (Q - self._qmin) / self._qspan if self.normalize_inputs else Q
        if self.input_axes is not None:
            X = X[:, self.input_axes]
        raw = model(*[X[:, i].unsqueeze(-1) for i in range(X.shape[1])])
        return self.output_offset + self.output_scale * raw

    def _derivative_scale(self, axes):
        """Characteristic derivative scale: u_scale / product of coordinate spans."""
        axis_index = {name: i for i, name in
                      enumerate(self.problem.physics.coordinate_names)}
        scale = self.output_scale
        for ax in axes:
            scale /= float(self._qspan[axis_index[ax]].item())
        return scale

    def _pde_residual(self, model, Q):
        """Compute derivatives by autograd, then call physics.residual."""
        Q = Q.clone().detach().requires_grad_(True)
        u = self._forward_physical(model, Q)
        d = self._derivatives(u, Q, self.problem.physics.required_derivatives())
        return self.problem.physics.residual(Q, u, d)

    # ------------------------------------------------------------------
    # Loss assembly.
    # ------------------------------------------------------------------
    def _compute_loss(self, model, Q_data, u_data):
        """loss = sum(IC) + sum(BC) + res_weight*mse_res + data_weight*mse_data.

        Q_data contains physical coordinates with shape (n, dim_query);
        u_data has shape (n,1).
        """
        mse = nn.MSELoss()

        # Initial conditions: derivative=() for values or (t,) for initial velocity.
        loss_ic = torch.tensor(0.0, device=self.device)
        for ic in self.ic_terms:
            Qc = ic["Q"].clone().detach().requires_grad_(True)
            u_pred = self._forward_physical(model, Qc)
            if ic["deriv"] == ():
                val = u_pred
                scale = self.output_scale
            else:
                # Differentiate IC predictions along the declared axes, e.g. (t,) gives u_t.
                val = self._derivatives(u_pred, Qc, {ic["deriv"]})[ic["deriv"]]
                scale = self._derivative_scale(ic["deriv"])
            loss_ic = loss_ic + mse(val / scale, ic["target"] / scale)

        # BC
        loss_bc = torch.tensor(0.0, device=self.device)
        for bc in self.bc_terms:
            if bc["kind"] == "dirichlet":
                Qc = bc["Q"]
                u_pred = self._forward_physical(model, Qc)
                loss_bc = loss_bc + mse(u_pred / self.output_scale,
                                        bc["target"] / self.output_scale)
            elif bc["kind"] == "neumann":
                Qc = bc["Q"].clone().detach().requires_grad_(True)
                u_pred = self._forward_physical(model, Qc)
                val = self._derivatives(u_pred, Qc, {bc["deriv"]})[bc["deriv"]]
                scale = self._derivative_scale(bc["deriv"])
                loss_bc = loss_bc + mse(val / scale, bc["target"] / scale)
            elif bc["kind"] == "periodic":
                Qa, Qb = bc["Qa"], bc["Qb"]
                ua = self._forward_physical(model, Qa)
                ub = self._forward_physical(model, Qb)
                loss_bc = loss_bc + mse(ua / self.output_scale,
                                        ub / self.output_scale)
            else:  # periodic_derivative
                Qa = bc["Qa"].clone().detach().requires_grad_(True)
                Qb = bc["Qb"].clone().detach().requires_grad_(True)
                ua = self._forward_physical(model, Qa)
                ub = self._forward_physical(model, Qb)
                der = {bc["deriv"]}
                ua_d = self._derivatives(ua, Qa, der)[bc["deriv"]]
                ub_d = self._derivatives(ub, Qb, der)[bc["deriv"]]
                dscale = self._derivative_scale(bc["deriv"])
                loss_bc = (loss_bc
                           + mse(ua / self.output_scale, ub / self.output_scale)
                           + mse(ua_d / dscale, ub_d / dscale))

        f_res = self._pde_residual(model, self.Q_res)
        mse_res = mse(f_res / self.residual_scale, torch.zeros_like(f_res))

        if Q_data.shape[0] == 0:
            mse_data = torch.tensor(0.0, device=self.device)
        else:
            u_pred = self._forward_physical(model, Q_data)
            mse_data = mse(u_pred / self.output_scale,
                           u_data / self.output_scale)

        total = (self.ic_weight * loss_ic + self.bc_weight * loss_bc
                 + self.res_weight * mse_res + self.data_weight * mse_data)
        self.last_loss_components = {
            "ic": float(loss_ic.detach().cpu()),
            "bc": float(loss_bc.detach().cpu()),
            "pde": float(mse_res.detach().cpu()),
            "data": float(mse_data.detach().cpu()),
            "total": float(total.detach().cpu()),
        }
        return total

    def fit(self, Q: np.ndarray, u: np.ndarray, **ctx):
        gen = ctx.get("gen", 0)
        maxgen = ctx.get("maxgen", 1)

        if self.lr_schedule == "exponential":
            self._set_lr_by_gen(gen, maxgen)
        elif self.lr_schedule != "none":
            raise KeyError(f"Unknown lr_schedule: {self.lr_schedule}")

        Q = np.asarray(Q, dtype=float).reshape(-1, self._dim)
        Q_data = torch.as_tensor(Q, dtype=torch.float32, device=self.device)
        u_data = torch.as_tensor(np.asarray(u, dtype=np.float32), dtype=torch.float32, device=self.device).reshape(-1, 1)

        self.model.train()
        for epoch in range(self.ever_epochs):
            self.optimizer.zero_grad()
            loss = self._compute_loss(self.model, Q_data, u_data)
            loss.backward()
            self.optimizer.step()

    def predict(self, Q: np.ndarray, batch_size: int = 65536) -> np.ndarray:
        Q = np.asarray(Q, dtype=float).reshape(-1, self._dim)
        N = Q.shape[0]
        pred = np.empty(N, dtype=float)
        self.model.eval()
        with torch.no_grad():
            for i in range(0, N, batch_size):
                j = min(i + batch_size, N)
                Qb = torch.tensor(Q[i:j], dtype=torch.float32, device=self.device)
                pred[i:j] = self._forward_physical(
                    self.model, Qb).detach().cpu().numpy().reshape(-1)
        return pred
