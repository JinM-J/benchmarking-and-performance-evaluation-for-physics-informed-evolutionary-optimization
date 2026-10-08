"""Physics-informed neural operator: FNO with field-level physics losses.

Following the operator-learning stage of PINO, one FNO backbone learns from
sparse HF labels and PDE/IC/BC constraints on denser regular grids. No
candidate-specific test-time optimization is performed, so surrogate fitness
queries do not trigger additional training.

Periodic axes use Fourier differentiation; nonperiodic axes use differentiable
finite differences. The protocol explicitly declares fourier_axes rather than
inferring periodicity from field shapes or data.

References:
- Li et al., Physics-Informed Neural Operator for Learning PDEs (2021)
  https://arxiv.org/abs/2111.03794
- Official implementation: https://github.com/neuraloperator/physics_informed
"""
from __future__ import annotations

import numpy as np
import torch

from surrogates.fno import FNOSurrogate


class PINOSurrogate(FNOSurrogate):
    """Field-level PINO with fit(Q,u) / predict(Q) methods."""

    name = "PINO"

    def __init__(self, config: dict | None = None):
        config = config or {}
        super().__init__(config)
        self.physics_instances = int(config.get("physics_instances", 8))
        self.ic_points = int(config.get("ic_points", 64))
        self.bc_points = int(config.get("bc_points", 64))
        self.obs_bc_points = int(config.get("obs_bc_points", self.bc_points))
        self.data_weight = float(config.get("data_weight", 5.0))
        self.res_weight = float(config.get("res_weight", 1.0))
        self.ic_weight = float(config.get("ic_weight", 1.0))
        self.bc_weight = float(config.get("bc_weight", 1.0))
        self.max_periodic_bc_order = config.get('max_periodic_bc_order', None)
        self.residual_scale = float(config.get("residual_scale", 1.0))
        self.fourier_axes = tuple(config.get("fourier_axes", ()))
        self.paired_physics_axes = self._parse_paired_physics_axes(
            config.get("paired_physics_axes", {})
        )
        self.residual_axis_margins = dict(
            config.get("residual_axis_margins", {})
        )
        self.last_loss_components = None

    @staticmethod
    def _parse_paired_physics_axes(config_value: dict) -> dict:
        if not isinstance(config_value, dict):
            raise TypeError('PINO paired_physics_axes must map axis names to value lists')
        return {
            str(axis_name): tuple(float(value) for value in values)
            for axis_name, values in config_value.items()
        }

    def _update_config(self, config: dict) -> None:
        super()._update_config(config)
        self.physics_instances = int(
            config.get("physics_instances", self.physics_instances)
        )
        self.ic_points = int(config.get("ic_points", self.ic_points))
        self.bc_points = int(config.get("bc_points", self.bc_points))
        self.obs_bc_points = int(
            config.get("obs_bc_points", self.obs_bc_points)
        )
        self.data_weight = float(config.get("data_weight", self.data_weight))
        self.res_weight = float(config.get("res_weight", self.res_weight))
        self.ic_weight = float(config.get("ic_weight", self.ic_weight))
        self.bc_weight = float(config.get("bc_weight", self.bc_weight))
        if 'max_periodic_bc_order' in config:
            self.max_periodic_bc_order = config['max_periodic_bc_order']
        self.residual_scale = float(
            config.get("residual_scale", self.residual_scale)
        )
        self.fourier_axes = tuple(
            config.get("fourier_axes", self.fourier_axes)
        )
        self.paired_physics_axes = self._parse_paired_physics_axes(
            config.get("paired_physics_axes", self.paired_physics_axes)
        )
        self.residual_axis_margins = dict(
            config.get("residual_axis_margins", self.residual_axis_margins)
        )

    def setup(self, problem, seed: int, config: dict = None):
        super().setup(problem, seed, config)
        if self.physics_instances < 1:
            raise ValueError("PINO physics_instances must be positive")
        if min(self.ic_points, self.bc_points, self.obs_bc_points) < 1:
            raise ValueError('PINO IC/BC point counts must be positive')
        if min(self.data_weight, self.res_weight,
               self.ic_weight, self.bc_weight) < 0.0:
            raise ValueError("PINO loss weights must be nonnegative")
        if self.residual_scale <= 0.0:
            raise ValueError("PINO residual_scale must be positive")

        if self.max_periodic_bc_order is not None:
            if isinstance(self.max_periodic_bc_order, (bool, np.bool_)) or not isinstance(self.max_periodic_bc_order, (int, np.integer)) or self.max_periodic_bc_order < 0:
                raise ValueError('PINO max_periodic_bc_order must be a nonnegative integer or None')
            self.max_periodic_bc_order = int(self.max_periodic_bc_order)
        coordinate_names = tuple(problem.physics.coordinate_names)
        if len(coordinate_names) != self._dim:
            raise ValueError(f'PINO physics.coordinate_names length {len(coordinate_names)} differs from query dimension {self._dim}')
        self._axis_to_query = {
            name: index for index, name in enumerate(coordinate_names)
        }
        self._axis_to_grid = {
            coordinate_names[query_axis]: grid_axis
            for grid_axis, query_axis in enumerate(self._grid_axes)
        }
        unknown_fourier = set(self.fourier_axes) - set(self._axis_to_grid)
        if unknown_fourier:
            raise ValueError(f'PINO fourier_axes includes nongrid axes: {sorted(unknown_fourier)}')
        for axis_name, values in self.paired_physics_axes.items():
            if axis_name not in self._axis_to_query:
                raise KeyError(f'PINO paired_physics_axes includes unknown axis {axis_name}')
            query_axis = self._axis_to_query[axis_name]
            if query_axis not in self._conditioning_axes:
                raise ValueError(
                    f"PINO paired axis {axis_name} must belong to conditioning_axes"
                )
            if not values:
                raise ValueError(f"PINO paired axis {axis_name} has no declared values")
            lower, upper = self.problem.query_bounds[query_axis]
            if any(value < lower or value > upper for value in values):
                raise ValueError(
                    f"PINO paired axis {axis_name} values must lie in [{lower}, {upper}]"
                )

        required = set(problem.physics.required_derivatives())
        for axes in required:
            if not axes or len(set(axes)) != 1:
                raise NotImplementedError(f'PINO supports repeated derivatives on a single axis; received {axes}')
            if axes[0] not in self._axis_to_grid:
                raise ValueError(
                    f"PINO derivative axis {axes[0]} is not in grid_axes={self._grid_axes}"
                )
            if len(axes) not in (1, 2, 3, 4):
                raise NotImplementedError(f'PINO supports derivative orders 1/2/3/4; received {axes}')
            grid_size = self.grid_shape[self._axis_to_grid[axes[0]]]
            minimum_points = 5 if len(axes) in (3, 4) else 3
            if axes[0] not in self.fourier_axes and grid_size < minimum_points:
                raise ValueError(f'PINO {axes} finite differences require at least {minimum_points} grid points')

        self._physics_parameter_rows = self._sample_physics_parameters()
        self._physics_inputs = self._make_grid_inputs(
            self._physics_parameter_rows
        )
        physics_Q_np = self._make_physics_grid(self._physics_parameter_rows)
        self._physics_Q = torch.as_tensor(
            physics_Q_np, dtype=torch.float32, device=self.device
        )
        residual_mask = self._make_residual_mask(physics_Q_np, required)
        if not np.any(residual_mask):
            raise ValueError('PINO PDE residual mask is empty; check grid, geometry and margin')
        self._residual_mask = torch.as_tensor(
            residual_mask, dtype=torch.bool, device=self.device
        )
        self._required_derivatives = required
        self._build_condition_terms()
        self.last_loss_components = None

    # ------------------------------------------------------------------
    # Physics instances and regular grids.
    # ------------------------------------------------------------------
    def _sample_physics_parameters(self) -> np.ndarray:
        """Sample conditioning parameters; expand declared paired axes by Cartesian products."""
        if self._n_parameters == 0:
            return np.empty((1, 0), dtype=np.float32)

        n = self.physics_instances
        queries = self.problem.sample_interior_queries(n, self.seed + 7919)
        if queries is None:
            rng = np.random.default_rng(self.seed + 7919)
            bounds = np.asarray(self.problem.query_bounds, dtype=float)
            queries = np.column_stack([
                rng.uniform(bounds[axis, 0], bounds[axis, 1], size=n)
                for axis in range(self._dim)
            ])
        queries = np.asarray(queries, dtype=np.float32).reshape(-1, self._dim)
        for axis_name, values in self.paired_physics_axes.items():
            query_axis = self._axis_to_query[axis_name]
            expanded = []
            for value in values:
                block = queries.copy()
                block[:, query_axis] = value
                expanded.append(block)
            queries = np.concatenate(expanded, axis=0)
        normalized = self._normalize_queries(queries)
        rows = np.asarray(
            normalized[:, self._conditioning_axes], dtype=np.float32
        )
        return np.unique(rows, axis=0)

    def _make_physics_grid(self, parameter_rows: np.ndarray) -> np.ndarray:
        """Return physical-coordinate grids of shape (B,N0,N1,dim_query)."""
        coordinate_grid = (
            self._coordinate_grid.detach().cpu().numpy().transpose(1, 2, 0)
        )
        n_groups = int(parameter_rows.shape[0])
        Q_normalized = np.zeros(
            (n_groups, *self.grid_shape, self._dim), dtype=np.float32
        )
        for grid_axis, query_axis in enumerate(self._grid_axes):
            Q_normalized[..., query_axis] = coordinate_grid[..., grid_axis]
        for condition_index, query_axis in enumerate(self._conditioning_axes):
            Q_normalized[..., query_axis] = parameter_rows[
                :, condition_index, None, None
            ]
        return (self._qmin_np + Q_normalized * self._qspan_np).astype(
            np.float32, copy=False
        )

    @staticmethod
    def _axis_stencil_valid(mask: np.ndarray, axis: int,
                            radius: int) -> np.ndarray:
        """Require radius neighbors on both sides to be valid; never difference across holes."""
        valid = np.ones_like(mask, dtype=bool)
        for offset in range(-radius, radius + 1):
            shifted = np.zeros_like(mask, dtype=bool)
            source = [slice(None)] * mask.ndim
            target = [slice(None)] * mask.ndim
            if offset > 0:
                source[axis] = slice(offset, None)
                target[axis] = slice(None, -offset)
            elif offset < 0:
                source[axis] = slice(None, offset)
                target[axis] = slice(-offset, None)
            shifted[tuple(target)] = mask[tuple(source)]
            valid &= shifted
        return valid

    def _make_residual_mask(self, physics_Q: np.ndarray,
                            required: set) -> np.ndarray:
        shape = physics_Q.shape[:-1]
        base_valid = np.asarray(
            self.problem.is_valid_query(physics_Q.reshape(-1, self._dim)),
            dtype=bool,
        ).reshape(shape)
        mask = base_valid.copy()

        for axes in required:
            axis_name = axes[0]
            array_axis = 1 + self._axis_to_grid[axis_name]  # Offset for the leading batch axis.
            if axis_name in self.fourier_axes:
                # The grid includes both endpoints; FFT uses only the first N-1 unique periodic points.
                endpoint = [slice(None)] * mask.ndim
                endpoint[array_axis] = -1
                mask[tuple(endpoint)] = False
            else:
                radius = 2 if len(axes) in (3, 4) else 1
                mask &= self._axis_stencil_valid(
                    base_valid, array_axis, radius
                )

        for axis_name, margins in self.residual_axis_margins.items():
            if axis_name not in self._axis_to_query:
                raise KeyError(f"Unknown residual_axis_margins axis: {axis_name}")
            if np.isscalar(margins):
                lower_margin = upper_margin = float(margins)
            else:
                lower_margin, upper_margin = map(float, margins)
            query_axis = self._axis_to_query[axis_name]
            lower = self._qmin_np[query_axis] + lower_margin
            upper = (self._qmin_np[query_axis] + self._qspan_np[query_axis]
                     - upper_margin)
            coordinates = physics_Q[..., query_axis]
            mask &= (coordinates >= lower) & (coordinates <= upper)
        return mask

    # ------------------------------------------------------------------
    # Apply all IC/BC constraints to the same batch of physics parameter instances.
    # ------------------------------------------------------------------
    def _expand_condition_queries(self, sampled_queries: np.ndarray):
        sampled_queries = np.asarray(
            sampled_queries, dtype=np.float32
        ).reshape(-1, self._dim)
        n_points = sampled_queries.shape[0]
        n_groups = self._physics_parameter_rows.shape[0]
        expanded = np.tile(sampled_queries, (n_groups, 1))
        for group_index, parameter_row in enumerate(
                self._physics_parameter_rows):
            block = slice(group_index * n_points, (group_index + 1) * n_points)
            for condition_index, query_axis in enumerate(
                    self._conditioning_axes):
                expanded[block, query_axis] = (
                    self._qmin_np[query_axis]
                    + parameter_row[condition_index] * self._qspan_np[query_axis]
                )
        normalized = self._normalize_queries(expanded)
        return {
            "Q": expanded,
            "query_grid": torch.as_tensor(
                normalized[:, self._grid_axes],
                dtype=torch.float32, device=self.device,
            ),
            "group_index": torch.arange(
                n_groups, device=self.device, dtype=torch.long
            ).repeat_interleave(n_points),
        }

    def _condition_term(self, region, n_points: int, target=None):
        term = self._expand_condition_queries(region.sample(n_points))
        if target is not None:
            target_values = np.asarray(
                target(term["Q"]), dtype=np.float32
            ).reshape(-1)
            term["target"] = torch.as_tensor(
                target_values, dtype=torch.float32, device=self.device
            )
        return term

    def _build_condition_terms(self) -> None:
        self._ic_terms = []
        for condition in self.problem.physics.initial_conditions():
            n_points = condition.n_points or self.ic_points
            term = self._condition_term(
                condition.region, n_points, condition.target
            )
            term["deriv"] = tuple(condition.derivative)
            self._ic_terms.append(term)

        budgets = {
            "bc_points": self.bc_points,
            "obs_bc_points": self.obs_bc_points,
        }
        self._bc_terms = []
        for condition in self.problem.physics.boundary_conditions():
            n_points = condition.n_points or budgets[condition.points_key]
            if condition.kind in ("dirichlet", "neumann"):
                term = self._condition_term(
                    condition.region, n_points, condition.target
                )
                term["kind"] = condition.kind
                term["deriv"] = tuple(condition.derivative)
            elif condition.kind in ("periodic", "periodic_derivative"):
                kind = condition.kind
                if kind == 'periodic_derivative' and self.max_periodic_bc_order is not None and (len(condition.derivative) > self.max_periodic_bc_order):
                    if not condition.include_value:
                        continue
                    kind = 'periodic'
                region_a, region_b = condition.region_pair
                term = {'kind': kind, 'a': self._condition_term(region_a, n_points), 'b': self._condition_term(region_b, n_points), 'deriv': tuple(condition.derivative), 'include_value': condition.include_value}
            else:
                raise KeyError(f"Unsupported PINO boundary type: {condition.kind}")
            self._bc_terms.append(term)

    # ------------------------------------------------------------------
    # Fourier and finite-difference derivatives.
    # ------------------------------------------------------------------
    def _fourier_derivative(self, field: torch.Tensor,
                            grid_axis: int, order: int) -> torch.Tensor:
        tensor_axis = 2 + grid_axis
        n_full = field.shape[tensor_axis]
        n_unique = n_full - 1
        spacing = (
            self._qspan_np[self._grid_axes[grid_axis]] / n_unique
        )
        core = field.narrow(tensor_axis, 0, n_unique)
        frequencies = torch.fft.fftfreq(
            n_unique, d=float(spacing), device=field.device
        )
        multiplier = (2j * torch.pi * frequencies) ** order
        multiplier_shape = [1] * field.ndim
        multiplier_shape[tensor_axis] = n_unique
        transformed = torch.fft.fft(core, dim=tensor_axis)
        derivative_core = torch.fft.ifft(
            transformed * multiplier.reshape(multiplier_shape),
            dim=tensor_axis,
        ).real
        # The final and first endpoints represent the same periodic location.
        return torch.cat([
            derivative_core,
            derivative_core.narrow(tensor_axis, 0, 1),
        ], dim=tensor_axis)

    @staticmethod
    def _slice(ndim: int, axis: int, value) -> tuple:
        index = [slice(None)] * ndim
        index[axis] = value
        return tuple(index)

    def _finite_difference(self, field: torch.Tensor, grid_axis: int,
                           order: int, include_boundary: bool) -> torch.Tensor:
        tensor_axis = 2 + grid_axis
        n_points = field.shape[tensor_axis]
        spacing = float(
            self._qspan_np[self._grid_axes[grid_axis]] / (n_points - 1)
        )
        derivative = torch.zeros_like(field)
        ndim = field.ndim

        if order == 1:
            center = self._slice(ndim, tensor_axis, slice(1, -1))
            plus = self._slice(ndim, tensor_axis, slice(2, None))
            minus = self._slice(ndim, tensor_axis, slice(None, -2))
            derivative[center] = (field[plus] - field[minus]) / (2.0 * spacing)
            if include_boundary:
                first = self._slice(ndim, tensor_axis, 0)
                second = self._slice(ndim, tensor_axis, 1)
                third = self._slice(ndim, tensor_axis, 2)
                last = self._slice(ndim, tensor_axis, -1)
                penultimate = self._slice(ndim, tensor_axis, -2)
                antepenultimate = self._slice(ndim, tensor_axis, -3)
                derivative[first] = (
                    -3.0 * field[first] + 4.0 * field[second] - field[third]
                ) / (2.0 * spacing)
                derivative[last] = (
                    3.0 * field[last] - 4.0 * field[penultimate]
                    + field[antepenultimate]
                ) / (2.0 * spacing)
            return derivative

        if order == 2:
            center = self._slice(ndim, tensor_axis, slice(1, -1))
            plus = self._slice(ndim, tensor_axis, slice(2, None))
            minus = self._slice(ndim, tensor_axis, slice(None, -2))
            derivative[center] = (field[plus] - 2.0 * field[center] + field[minus]) / spacing ** 2
            if include_boundary:
                if n_points < 4:
                    raise ValueError('Boundary second derivatives require four grid points')
                for position, indices in ((0, (0, 1, 2, 3)), (-1, (-1, -2, -3, -4))):
                    values = [field[self._slice(ndim, tensor_axis, k)] for k in indices]
                    derivative[self._slice(ndim, tensor_axis, position)] = (2.0 * values[0] - 5.0 * values[1] + 4.0 * values[2] - values[3]) / spacing ** 2
            return derivative
        if order == 3:
            if n_points < 5:
                raise ValueError('Third derivatives require five grid points')
            center = self._slice(ndim, tensor_axis, slice(2, -2))
            plus_one = self._slice(ndim, tensor_axis, slice(3, -1))
            plus_two = self._slice(ndim, tensor_axis, slice(4, None))
            minus_one = self._slice(ndim, tensor_axis, slice(1, -3))
            minus_two = self._slice(ndim, tensor_axis, slice(None, -4))
            derivative[center] = (field[plus_two] - 2.0 * field[plus_one] + 2.0 * field[minus_one] - field[minus_two]) / (2.0 * spacing ** 3)
            if include_boundary:
                stencils = ((0, (0, 1, 2, 3, 4), (-5, 18, -24, 14, -3)), (1, (0, 1, 2, 3, 4), (-3, 10, -12, 6, -1)), (-2, (-1, -2, -3, -4, -5), (3, -10, 12, -6, 1)), (-1, (-1, -2, -3, -4, -5), (5, -18, 24, -14, 3)))
                for position, indices, weights in stencils:
                    value = sum((w * field[self._slice(ndim, tensor_axis, k)] for k, w in zip(indices, weights)))
                    derivative[self._slice(ndim, tensor_axis, position)] = value / (2.0 * spacing ** 3)
            return derivative

        if order == 4:
            center = self._slice(ndim, tensor_axis, slice(2, -2))
            plus_one = self._slice(ndim, tensor_axis, slice(3, -1))
            plus_two = self._slice(ndim, tensor_axis, slice(4, None))
            minus_one = self._slice(ndim, tensor_axis, slice(1, -3))
            minus_two = self._slice(ndim, tensor_axis, slice(None, -4))
            derivative[center] = (
                field[minus_two] - 4.0 * field[minus_one]
                + 6.0 * field[center] - 4.0 * field[plus_one]
                + field[plus_two]
            ) / spacing ** 4
            return derivative

        raise NotImplementedError(f'PINO does not support {order}-order finite differences')

    def _field_derivative(self, field: torch.Tensor, axes: tuple,
                          include_boundary: bool = False) -> torch.Tensor:
        if not axes or len(set(axes)) != 1:
            raise NotImplementedError(f"PINO does not support derivative {axes}")
        axis_name = axes[0]
        grid_axis = self._axis_to_grid[axis_name]
        order = len(axes)
        if axis_name in self.fourier_axes:
            return self._fourier_derivative(field, grid_axis, order)
        return self._finite_difference(
            field, grid_axis, order, include_boundary
        )

    def _derivative_scale(self, axes: tuple) -> float:
        query_axis = self._axis_to_query[axes[0]]
        return self.output_scale / float(self._qspan_np[query_axis]) ** len(axes)

    def _sample_term(self, field: torch.Tensor, term: dict) -> torch.Tensor:
        return self._sample_field(
            field, term["query_grid"], term["group_index"]
        )

    # ------------------------------------------------------------------
    # Backpropagate data loss, then PDE/IC/BC loss within the same optimizer step.
    # ------------------------------------------------------------------
    def _physics_losses(self):
        raw_field = self.model(self._physics_inputs)
        field = self.output_offset + self.output_scale * raw_field
        derivative_cache = {}

        def derivative(axes, include_boundary=False):
            key = (tuple(axes), bool(include_boundary))
            if key not in derivative_cache:
                derivative_cache[key] = self._field_derivative(
                    field, tuple(axes), include_boundary
                )
            return derivative_cache[key]

        Q_flat = self._physics_Q.reshape(-1, self._dim)
        u_flat = field[:, 0].reshape(-1, 1)
        derivative_values = {
            axes: derivative(axes, False)[:, 0].reshape(-1, 1)
            for axes in self._required_derivatives
        }
        residual = self.problem.physics.residual(
            Q_flat, u_flat, derivative_values
        ).reshape(-1)
        residual_values = residual[self._residual_mask.reshape(-1)]
        loss_pde = torch.mean((residual_values / self.residual_scale) ** 2)

        loss_ic = torch.zeros((), dtype=field.dtype, device=self.device)
        for term in self._ic_terms:
            if term["deriv"]:
                values = self._sample_term(
                    derivative(term["deriv"], True), term
                )
                scale = self._derivative_scale(term["deriv"])
            else:
                values = self._sample_term(field, term)
                scale = self.output_scale
            loss_ic = loss_ic + torch.mean(
                ((values - term["target"]) / scale) ** 2
            )

        loss_bc = torch.zeros((), dtype=field.dtype, device=self.device)
        for term in self._bc_terms:
            if term["kind"] == "dirichlet":
                values = self._sample_term(field, term)
                loss_bc = loss_bc + torch.mean(
                    ((values - term["target"]) / self.output_scale) ** 2
                )
            elif term["kind"] == "neumann":
                derivative_field = derivative(term["deriv"], True)
                values = self._sample_term(derivative_field, term)
                scale = self._derivative_scale(term["deriv"])
                loss_bc = loss_bc + torch.mean(
                    ((values - term["target"]) / scale) ** 2
                )
            else:
                values_a = self._sample_term(field, term["a"])
                values_b = self._sample_term(field, term["b"])
                if term.get('include_value', True):
                    loss_bc = loss_bc + torch.mean(((values_a - values_b) / self.output_scale) ** 2)
                if term["kind"] == "periodic_derivative":
                    derivative_field = derivative(term["deriv"], True)
                    derivative_a = self._sample_term(
                        derivative_field, term["a"]
                    )
                    derivative_b = self._sample_term(
                        derivative_field, term["b"]
                    )
                    scale = self._derivative_scale(term["deriv"])
                    loss_bc = loss_bc + torch.mean(
                        ((derivative_a - derivative_b) / scale) ** 2
                    )
        return loss_ic, loss_bc, loss_pde

    def _pde_residual(self, model, Q: torch.Tensor) -> torch.Tensor:
        """Evaluate grid residuals at scattered Q; invalid finite-difference stencils yield NaN.

        PINO differentiates its output grid, not query coordinates via autograd.
        Generate fields and residual grids per conditioning vector, then sample
        bilinearly at queries. Incomplete stencils near outer boundaries or holes
        produce nonfinite values for metrics to skip, never artificial zero derivatives.
        """
        Q_np = np.asarray(
            Q.detach().cpu(), dtype=np.float32
        ).reshape(-1, self._dim)
        if Q_np.shape[0] == 0:
            return torch.empty((0, 1), dtype=torch.float32, device=self.device)
        normalized = self._normalize_queries(Q_np)
        parameter_rows, inverse = self._group_parameters(normalized)
        output = torch.full(
            (Q_np.shape[0],), float("nan"),
            dtype=torch.float32, device=self.device,
        )

        with torch.no_grad():
            for start in range(0, parameter_rows.shape[0],
                               self.parameter_batch_size):
                stop = min(start + self.parameter_batch_size,
                           parameter_rows.shape[0])
                point_indices = np.flatnonzero(
                    (inverse >= start) & (inverse < stop)
                )
                local_group = torch.as_tensor(
                    inverse[point_indices] - start,
                    dtype=torch.long, device=self.device,
                )
                query_grid = torch.as_tensor(
                    normalized[np.ix_(point_indices, self._grid_axes)],
                    dtype=torch.float32, device=self.device,
                )
                rows = parameter_rows[start:stop]
                raw_field = model(self._make_grid_inputs(rows))
                field = self.output_offset + self.output_scale * raw_field
                physics_Q_np = self._make_physics_grid(rows)
                physics_Q = torch.as_tensor(
                    physics_Q_np, dtype=torch.float32, device=self.device
                )
                derivatives = {
                    axes: self._field_derivative(field, axes, False)[:, 0]
                    .reshape(-1, 1)
                    for axes in self._required_derivatives
                }
                residual = self.problem.physics.residual(
                    physics_Q.reshape(-1, self._dim),
                    field[:, 0].reshape(-1, 1),
                    derivatives,
                ).reshape(len(rows), 1, *self.grid_shape)
                valid_np = self._make_residual_mask(
                    physics_Q_np, self._required_derivatives
                )
                valid = torch.as_tensor(
                    valid_np, dtype=torch.bool, device=self.device
                ).unsqueeze(1)
                safe_residual = torch.where(valid, residual,
                                            torch.zeros_like(residual))
                sampled = self._sample_field(
                    safe_residual, query_grid, local_group
                )
                sampled_valid = self._sample_field(
                    valid.to(field.dtype), query_grid, local_group
                ) > (1.0 - 1.0e-6)
                sampled = torch.where(
                    sampled_valid, sampled,
                    torch.full_like(sampled, float("nan")),
                )
                output[torch.as_tensor(
                    point_indices, dtype=torch.long, device=self.device
                )] = sampled
        return output.reshape(-1, 1)

    def fit(self, Q: np.ndarray, u: np.ndarray, **ctx):
        if self.model is None:
            raise RuntimeError('PINO setup has not been called')
        Q = np.asarray(Q, dtype=np.float32).reshape(-1, self._dim)
        targets = np.asarray(u, dtype=np.float32).reshape(-1)
        if Q.shape[0] != targets.shape[0]:
            raise ValueError('PINO Q and u sample counts differ')
        if not np.all(np.isfinite(Q)) or not np.all(np.isfinite(targets)):
            raise ValueError("PINO training data contains nonfinite values")

        if self.lr_schedule == "exponential":
            self._set_lr_by_gen(ctx.get("gen", 0), ctx.get("maxgen", 1))
        elif self.lr_schedule != "none":
            raise KeyError(f"Unknown PINO lr_schedule: {self.lr_schedule}")

        if Q.shape[0]:
            normalized = self._normalize_queries(Q)
            parameter_rows, inverse = self._group_parameters(normalized)
            data_batches = self._make_parameter_batches(
                normalized, parameter_rows, inverse, targets
            )
        else:
            data_batches = []
        n_data = float(max(1, Q.shape[0]))

        self.model.train()
        for _ in range(self.ever_epochs):
            self.optimizer.zero_grad()
            data_value = torch.zeros((), device=self.device)
            for batch in data_batches:
                raw_field = self.model(batch["inputs"])
                prediction = self._sample_field(
                    raw_field, batch["query_xt"], batch["group_index"]
                )
                batch_loss = torch.sum(
                    (prediction - batch["targets"]) ** 2
                ) / n_data
                (self.data_weight * batch_loss).backward()
                data_value = data_value + batch_loss.detach()

            loss_ic, loss_bc, loss_pde = self._physics_losses()
            physics_total = (
                self.ic_weight * loss_ic
                + self.bc_weight * loss_bc
                + self.res_weight * loss_pde
            )
            physics_total.backward()
            self.optimizer.step()

            total_value = (
                self.data_weight * data_value
                + self.ic_weight * loss_ic.detach()
                + self.bc_weight * loss_bc.detach()
                + self.res_weight * loss_pde.detach()
            )
            self.last_loss_components = {
                "ic": float(loss_ic.detach().cpu()),
                "bc": float(loss_bc.detach().cpu()),
                "pde": float(loss_pde.detach().cpu()),
                "data": float(data_value.cpu()),
                "total": float(total_value.cpu()),
            }
