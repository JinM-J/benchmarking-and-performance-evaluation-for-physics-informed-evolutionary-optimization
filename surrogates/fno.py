"""FNO field surrogate: regular-grid spectral network trained on sparse HF labels.

The external Surrogate interface accepts scattered Q -> u pairs. Internally,
FNO generates a full (x,t) grid and samples queries bilinearly, preserving the
HF pool, FE accounting and objective/constraint semantics.

The first two query axes form the grid by default; remaining axes are broadcast
conditioning channels. grid_axes can select other axes when a problem puts
parameters before physical coordinates. conditioning_axes optionally selects
only the axes affecting the PDE; otherwise all non-grid axes condition it.
domain_padding appends zeros along nonperiodic axes before convolution and
crops afterward; [0,0] preserves the unpadded behavior.

This data-only baseline never accesses Problem.physics or adds PDE/IC/BC losses.
The physics-informed counterpart is PINO. Spectral layers follow Li et al.
(ICLR 2021): rfft2, learned complex weights on low positive/negative x modes,
and irfft2. Reference implementation:
https://github.com/neuraloperator/neuraloperator
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from surrogates.base import Surrogate


class SpectralConv2d(nn.Module):
    """2D Fourier convolution with input/output shape (B,C,Nx,Nt)."""

    def __init__(self, in_channels: int, out_channels: int,
                 modes_x: int, modes_t: int):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.modes_x = int(modes_x)
        self.modes_t = int(modes_t)

        # Separate positive/negative x modes; rFFT stores only nonnegative t frequencies.
        scale = (2.0 / (self.in_channels + self.out_channels)) ** 0.5
        shape = (self.in_channels, self.out_channels,
                 self.modes_x, self.modes_t)
        self.weight_pos = nn.Parameter(
            scale * torch.randn(*shape, dtype=torch.cfloat)
        )
        self.weight_neg = nn.Parameter(
            scale * torch.randn(*shape, dtype=torch.cfloat)
        )

    @staticmethod
    def _contract(x_ft: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        # (B,Cin,mx,mt) × (Cin,Cout,mx,mt) -> (B,Cout,mx,mt)
        return torch.einsum("bixy,ioxy->boxy", x_ft, weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _, n_x, n_t = x.shape
        if 2 * self.modes_x > n_x:
            raise ValueError(
                f"FNO modes_x={self.modes_x} requires 2*modes_x <= n_x={n_x}"
            )
        n_t_rfft = n_t // 2 + 1
        if self.modes_t > n_t_rfft:
            raise ValueError(
                f"FNO modes_t={self.modes_t} exceeds the rFFT limit {n_t_rfft}"
            )

        x_ft = torch.fft.rfft2(x, norm="ortho")
        out_ft = torch.zeros(
            batch_size, self.out_channels, n_x, n_t_rfft,
            dtype=torch.cfloat, device=x.device,
        )
        out_ft[:, :, :self.modes_x, :self.modes_t] = self._contract(
            x_ft[:, :, :self.modes_x, :self.modes_t], self.weight_pos
        )
        out_ft[:, :, -self.modes_x:, :self.modes_t] = self._contract(
            x_ft[:, :, -self.modes_x:, :self.modes_t], self.weight_neg
        )
        return torch.fft.irfft2(out_ft, s=(n_x, n_t), norm="ortho")


class FNO2d(nn.Module):
    """lifting -> Fourier blocks -> pointwise projection。"""

    def __init__(self, in_channels: int, hidden_channels: int,
                 modes_x: int, modes_t: int, n_layers: int,
                 projection_channels: int,
                 domain_padding: tuple[int, int] = (0, 0)):
        super().__init__()
        self.domain_padding = tuple(int(value) for value in domain_padding)
        if len(self.domain_padding) != 2 or min(self.domain_padding) < 0:
            raise ValueError(
                "FNO domain_padding must contain two nonnegative integers"
            )
        self.lifting = nn.Conv2d(in_channels, hidden_channels, kernel_size=1)
        self.spectral_layers = nn.ModuleList([
            SpectralConv2d(hidden_channels, hidden_channels, modes_x, modes_t)
            for _ in range(n_layers)
        ])
        # FNO block: global spectral convolution plus a local pointwise linear branch.
        self.local_layers = nn.ModuleList([
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1)
            for _ in range(n_layers)
        ])
        self.projection_in = nn.Conv2d(
            hidden_channels, projection_channels, kernel_size=1
        )
        self.projection_out = nn.Conv2d(projection_channels, 1, kernel_size=1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        field = self.lifting(inputs)
        n_x, n_t = field.shape[-2:]
        pad_x, pad_t = self.domain_padding
        if pad_x or pad_t:
            # Pad the high end of nonperiodic axes to separate physical boundaries in the FFT.
            field = F.pad(field, (0, pad_t, 0, pad_x))
        for layer_index, (spectral, local) in enumerate(
                zip(self.spectral_layers, self.local_layers)):
            field = spectral(field) + local(field)
            if layer_index + 1 < len(self.spectral_layers):
                field = F.gelu(field)
        if pad_x or pad_t:
            field = field[..., :n_x, :n_t]
        return self.projection_out(F.gelu(self.projection_in(field)))


class FNOSurrogate(Surrogate):
    """FNO field surrogate exposing scattered-point fit/predict methods."""

    name = "FNO"

    def __init__(self, config: dict | None = None):
        config = config or {}
        self.grid_shape = tuple(config.get("grid_shape", [64, 64]))
        self.n_modes = tuple(config.get("n_modes", [12, 12]))
        self.hidden_channels = int(config.get("hidden_channels", 32))
        self.n_layers = int(config.get("n_layers", 4))
        self.projection_channels = int(config.get("projection_channels", 64))
        self.ever_epochs = int(config.get("ever_epochs", 300))
        self.lr = float(config.get("lr", 1.0e-3))
        self.lr_max = float(config.get("lr_max", self.lr))
        self.lr_min = float(config.get("lr_min", 1.0e-5))
        self.lr_schedule = config.get("lr_schedule", "exponential")
        self.parameter_batch_size = int(config.get("parameter_batch_size", 16))
        self.seed_model_init = bool(config.get("seed_model_init", True))
        # Preserve default weights/RNG; small-weight initialization requires an explicit protocol.
        self.spectral_initialization = config.get("spectral_initialization", "legacy")
        grid_axes = config.get("grid_axes", None)
        self.grid_axes = None if grid_axes is None else tuple(int(a) for a in grid_axes)
        conditioning_axes = config.get("conditioning_axes", None)
        self.conditioning_axes = (
            None if conditioning_axes is None
            else tuple(int(a) for a in conditioning_axes)
        )
        self.domain_padding = tuple(config.get("domain_padding", [0, 0]))
        self.output_offset = float(config.get("output_offset", 0.0))
        self.output_scale = float(config.get("output_scale", 1.0))

        self.model = None
        self.optimizer = None
        self.problem = None

    def _update_config(self, config: dict) -> None:
        self.grid_shape = tuple(config.get("grid_shape", self.grid_shape))
        self.n_modes = tuple(config.get("n_modes", self.n_modes))
        self.hidden_channels = int(config.get("hidden_channels", self.hidden_channels))
        self.n_layers = int(config.get("n_layers", self.n_layers))
        self.projection_channels = int(
            config.get("projection_channels", self.projection_channels)
        )
        self.ever_epochs = int(config.get("ever_epochs", self.ever_epochs))
        self.lr = float(config.get("lr", self.lr))
        self.lr_max = float(config.get("lr_max", self.lr_max))
        self.lr_min = float(config.get("lr_min", self.lr_min))
        self.lr_schedule = config.get("lr_schedule", self.lr_schedule)
        self.parameter_batch_size = int(
            config.get("parameter_batch_size", self.parameter_batch_size)
        )
        self.seed_model_init = bool(
            config.get("seed_model_init", self.seed_model_init)
        )
        self.spectral_initialization = config.get(
            "spectral_initialization", self.spectral_initialization
        )
        grid_axes = config.get("grid_axes", self.grid_axes)
        self.grid_axes = None if grid_axes is None else tuple(int(a) for a in grid_axes)
        conditioning_axes = config.get("conditioning_axes", self.conditioning_axes)
        self.conditioning_axes = (
            None if conditioning_axes is None
            else tuple(int(a) for a in conditioning_axes)
        )
        self.domain_padding = tuple(
            config.get("domain_padding", self.domain_padding)
        )
        self.output_offset = float(config.get("output_offset", self.output_offset))
        self.output_scale = float(config.get("output_scale", self.output_scale))

    def setup(self, problem, seed: int, config: dict = None):
        if config:
            self._update_config(config)
        if len(self.grid_shape) != 2 or min(self.grid_shape) < 2:
            raise ValueError(f"FNO grid_shape must contain two integers >=2; received {self.grid_shape}")
        if len(self.n_modes) != 2 or min(self.n_modes) < 1:
            raise ValueError(f"FNO n_modes must contain two positive integers; received {self.n_modes}")
        if 2 * self.n_modes[0] > self.grid_shape[0]:
            raise ValueError("FNO requires 2*n_modes[0] <= grid_shape[0]")
        if self.n_modes[1] > self.grid_shape[1] // 2 + 1:
            raise ValueError("FNO n_modes[1] exceeds the grid's rFFT limit")
        if self.hidden_channels < 1 or self.n_layers < 1:
            raise ValueError("FNO hidden_channels and n_layers must be positive")
        if self.projection_channels < 1 or self.parameter_batch_size < 1:
            raise ValueError("FNO projection_channels and parameter_batch_size must be positive")
        if self.lr <= 0.0 or self.lr_max <= 0.0 or self.lr_min <= 0.0:
            raise ValueError("FNO learning rate must be positive")
        if self.output_scale <= 0.0:
            raise ValueError("FNO output_scale must be positive")
        if self.spectral_initialization not in ("legacy", "official_uniform"):
            raise ValueError("spectral_initialization must be legacy or official_uniform")
        if len(self.domain_padding) != 2 or any(
                int(value) != value or int(value) < 0
                for value in self.domain_padding):
            raise ValueError("FNO domain_padding must contain two nonnegative integers")
        self.domain_padding = tuple(int(value) for value in self.domain_padding)

        self.problem = problem
        self.seed = int(seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        query_bounds = np.asarray(problem.query_bounds, dtype=np.float32)
        self._dim = int(query_bounds.shape[0])
        if self._dim < 2:
            raise ValueError("FNO requires at least two spatial/temporal query axes")
        self._grid_axes = self.grid_axes or (0, 1)
        if (len(self._grid_axes) != 2
                or len(set(self._grid_axes)) != 2
                or min(self._grid_axes) < 0
                or max(self._grid_axes) >= self._dim):
            raise ValueError(
                f"FNO grid_axes={self._grid_axes} is invalid for a {self._dim}-dimensional query space"
            )
        if self.conditioning_axes is None:
            self._conditioning_axes = tuple(
                axis for axis in range(self._dim) if axis not in self._grid_axes
            )
        else:
            self._conditioning_axes = self.conditioning_axes
            if (len(set(self._conditioning_axes)) != len(self._conditioning_axes)
                    or any(axis < 0 or axis >= self._dim
                           for axis in self._conditioning_axes)
                    or set(self._conditioning_axes) & set(self._grid_axes)):
                raise ValueError(
                    "FNO conditioning_axes must be unique, valid and disjoint from grid_axes; "
                    f"received {self._conditioning_axes}"
                )
        self._n_parameters = len(self._conditioning_axes)
        self._qmin_np = query_bounds[:, 0]
        self._qspan_np = query_bounds[:, 1] - query_bounds[:, 0]
        if np.any(self._qspan_np <= 0.0):
            raise ValueError("FNO query_bounds must have a positive span on every axis")

        if self.seed_model_init:
            torch.manual_seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.seed)

        n_x, n_t = self.grid_shape
        x_grid = torch.linspace(0.0, 1.0, n_x, device=self.device)
        t_grid = torch.linspace(0.0, 1.0, n_t, device=self.device)
        X, T = torch.meshgrid(x_grid, t_grid, indexing="ij")
        self._coordinate_grid = torch.stack([X, T], dim=0)  # (2,Nx,Nt)

        self.model = FNO2d(
            in_channels=2 + self._n_parameters,
            hidden_channels=self.hidden_channels,
            modes_x=self.n_modes[0],
            modes_t=self.n_modes[1],
            n_layers=self.n_layers,
            projection_channels=self.projection_channels,
            domain_padding=self.domain_padding,
        ).to(self.device)
        if self.spectral_initialization == "official_uniform":
            # Complex uniform/(Cin*Cout) initialization from physics_informed/models/basics.py.
            # Replace spectral weights after constructing the network; retain local/projection weights.
            # Use an independent CPU RNG for reproducible ablation and replay.
            generator = torch.Generator(device="cpu").manual_seed(self.seed)
            with torch.no_grad():
                for layer in self.model.spectral_layers:
                    scale = 1.0 / (layer.in_channels * layer.out_channels)
                    for weight in (layer.weight_pos, layer.weight_neg):
                        initial = scale * torch.rand(
                            weight.shape, dtype=weight.dtype, generator=generator
                        )
                        weight.copy_(initial.to(weight.device))
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def _set_lr_by_gen(self, gen: int, maxgen: int) -> float:
        generation = float(np.clip(gen, 0, maxgen))
        lr_now = self.lr_max * (
            (self.lr_min / self.lr_max) ** (generation / max(1, maxgen))
        )
        for group in self.optimizer.param_groups:
            group["lr"] = float(lr_now)
        return float(lr_now)

    def _normalize_queries(self, Q: np.ndarray) -> np.ndarray:
        return np.clip((Q - self._qmin_np) / self._qspan_np, 0.0, 1.0)

    def _group_parameters(self, Q_normalized: np.ndarray):
        """Group by conditioning vector; an unconditional 2D field has one empty group."""
        n_points = Q_normalized.shape[0]
        if self._n_parameters == 0:
            return np.empty((1, 0), dtype=np.float32), np.zeros(n_points, dtype=int)
        parameter_rows = np.asarray(
            Q_normalized[:, self._conditioning_axes], dtype=np.float32
        )
        return np.unique(parameter_rows, axis=0, return_inverse=True)

    def _make_grid_inputs(self, parameter_rows: np.ndarray) -> torch.Tensor:
        n_groups = int(parameter_rows.shape[0])
        coordinate_grid = self._coordinate_grid.unsqueeze(0).expand(
            n_groups, -1, -1, -1
        )
        if self._n_parameters == 0:
            return coordinate_grid
        parameters = torch.as_tensor(
            parameter_rows, dtype=torch.float32, device=self.device
        ).view(n_groups, self._n_parameters, 1, 1)
        parameters = parameters.expand(-1, -1, *self.grid_shape)
        return torch.cat([coordinate_grid, parameters], dim=1)

    @staticmethod
    def _sample_field(field: torch.Tensor, query_xt: torch.Tensor,
                      group_index: torch.Tensor) -> torch.Tensor:
        """Vectorized bilinear sampling from (Nx,Nt) fields; return shape (N,)."""
        n_x, n_t = field.shape[-2:]
        x_pos = query_xt[:, 0] * (n_x - 1)
        t_pos = query_xt[:, 1] * (n_t - 1)
        x0 = torch.floor(x_pos).long().clamp(0, n_x - 1)
        t0 = torch.floor(t_pos).long().clamp(0, n_t - 1)
        x1 = (x0 + 1).clamp(max=n_x - 1)
        t1 = (t0 + 1).clamp(max=n_t - 1)
        wx = x_pos - x0.to(x_pos.dtype)
        wt = t_pos - t0.to(t_pos.dtype)

        values = field[:, 0]
        v00 = values[group_index, x0, t0]
        v10 = values[group_index, x1, t0]
        v01 = values[group_index, x0, t1]
        v11 = values[group_index, x1, t1]
        return ((1.0 - wx) * (1.0 - wt) * v00
                + wx * (1.0 - wt) * v10
                + (1.0 - wx) * wt * v01
                + wx * wt * v11)

    def _make_parameter_batches(self, Q_normalized: np.ndarray,
                                parameter_rows: np.ndarray,
                                inverse: np.ndarray,
                                targets: np.ndarray | None = None):
        batches = []
        n_groups = parameter_rows.shape[0]
        for start in range(0, n_groups, self.parameter_batch_size):
            stop = min(start + self.parameter_batch_size, n_groups)
            point_indices = np.flatnonzero((inverse >= start) & (inverse < stop))
            batches.append({
                "point_indices": point_indices,
                "inputs": self._make_grid_inputs(parameter_rows[start:stop]),
                "query_xt": torch.as_tensor(
                    Q_normalized[np.ix_(point_indices, self._grid_axes)],
                    dtype=torch.float32, device=self.device,
                ),
                "group_index": torch.as_tensor(
                    inverse[point_indices] - start,
                    dtype=torch.long, device=self.device,
                ),
                "targets": None if targets is None else torch.as_tensor(
                    (targets[point_indices] - self.output_offset) / self.output_scale,
                    dtype=torch.float32, device=self.device,
                ),
            })
        return batches

    def fit(self, Q: np.ndarray, u: np.ndarray, **ctx):
        if self.model is None:
            raise RuntimeError("FNO has not been set up")
        Q = np.asarray(Q, dtype=np.float32).reshape(-1, self._dim)
        targets = np.asarray(u, dtype=np.float32).reshape(-1)
        if Q.shape[0] != targets.shape[0]:
            raise ValueError("FNO Q and u have different sample counts")
        if Q.shape[0] == 0:
            return
        if not np.all(np.isfinite(Q)) or not np.all(np.isfinite(targets)):
            raise ValueError("FNO training data contains nonfinite values")

        if self.lr_schedule == "exponential":
            self._set_lr_by_gen(ctx.get("gen", 0), ctx.get("maxgen", 1))
        elif self.lr_schedule != "none":
            raise KeyError(f"Unknown FNO lr_schedule: {self.lr_schedule}")

        Q_normalized = self._normalize_queries(Q)
        parameter_rows, inverse = self._group_parameters(Q_normalized)
        batches = self._make_parameter_batches(
            Q_normalized, parameter_rows, inverse, targets
        )
        n_total = float(Q.shape[0])

        self.model.train()
        for _ in range(self.ever_epochs):
            self.optimizer.zero_grad()
            for batch in batches:
                field = self.model(batch["inputs"])
                prediction = self._sample_field(
                    field, batch["query_xt"], batch["group_index"]
                )
                # Normalize each parameter minibatch SSE by total point count to accumulate full-pool MSE.
                loss = torch.sum((prediction - batch["targets"]) ** 2) / n_total
                loss.backward()
            self.optimizer.step()

    def predict(self, Q: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("FNO has not been set up")
        Q = np.asarray(Q, dtype=np.float32).reshape(-1, self._dim)
        if Q.shape[0] == 0:
            return np.empty((0,), dtype=float)
        Q_normalized = self._normalize_queries(Q)
        parameter_rows, inverse = self._group_parameters(Q_normalized)
        batches = self._make_parameter_batches(Q_normalized, parameter_rows, inverse)
        prediction = np.empty(Q.shape[0], dtype=float)

        self.model.eval()
        with torch.no_grad():
            for batch in batches:
                field = self.model(batch["inputs"])
                values = self._sample_field(
                    field, batch["query_xt"], batch["group_index"]
                )
                values = self.output_offset + self.output_scale * values
                prediction[batch["point_indices"]] = values.detach().cpu().numpy()
        return prediction
