"""Generate F07 Poisson reference data on a perforated domain."""

import argparse
import os
from pathlib import Path

import numpy as np


DEFAULT_OUTPUT = Path(__file__).resolve().parent / "f07.npz"


def solve_perforated_poisson(
    n_grid=1000,
    length=5.0,
    amplitude=20.0,
    mu1=2.0,
    mu2=4.0,
    stencil="9pt",
    cg_rtol=1e-10,
    cg_maxiter=20000,
):
    """Solve perforated-domain Poisson; return grid, state, source, and hole mask."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.linalg import LinearOperator, cg

    spacing = length / (n_grid - 1)
    spacing_squared = spacing * spacing
    x = np.linspace(0.0, length, n_grid)
    y = np.linspace(0.0, length, n_grid)
    x_grid, y_grid = np.meshgrid(x, y)
    source = amplitude * (
        mu1 ** 2 + mu2 ** 2 + x_grid ** 2 + y_grid ** 2
    ) * np.sin(mu1 * np.pi * x_grid) * np.sin(mu2 * np.pi * y_grid)

    u = np.zeros((n_grid, n_grid), dtype=float)
    outer_boundary_value = 0.2
    u[:, 0] = outer_boundary_value
    u[:, -1] = outer_boundary_value
    u[0, :] = outer_boundary_value
    u[-1, :] = outer_boundary_value

    circles = (
        ((4.3, 3.5), 0.5),
        ((1.5, 1.2), 0.4),
        ((2.2, 2.7), 0.6),
        ((0.6, 0.5), 0.3),
    )
    circle_inside = np.zeros((n_grid, n_grid), dtype=bool)
    circle_boundary = np.zeros((n_grid, n_grid), dtype=bool)
    for (center_x, center_y), radius in circles:
        distance_squared = (
            (x_grid - center_x) ** 2 + (y_grid - center_y) ** 2
        )
        distance = np.sqrt(distance_squared)
        circle_inside |= distance_squared < radius ** 2
        circle_boundary |= np.abs(distance - radius) < 0.75 * spacing

    fixed_mask = circle_inside | circle_boundary
    u[fixed_mask] = 1.0
    update_mask = np.zeros((n_grid, n_grid), dtype=bool)
    update_mask[1:-1, 1:-1] = True
    update_mask &= ~fixed_mask

    unknown_id = -np.ones((n_grid, n_grid), dtype=int)
    unknown_flat = np.flatnonzero(update_mask.ravel())
    unknown_id.ravel()[unknown_flat] = np.arange(unknown_flat.size)
    n_unknowns = unknown_flat.size
    if n_unknowns == 0:
        raise ValueError("F07 has no interior unknowns; check the hole mask")
    row_index, column_index = np.unravel_index(
        unknown_flat, (n_grid, n_grid)
    )

    if stencil == "5pt":
        diagonal = 4.0
        neighbors = (
            (1, 0, 1.0), (-1, 0, 1.0),
            (0, 1, 1.0), (0, -1, 1.0),
        )
        rhs_factor = -spacing_squared
    elif stencil == "9pt":
        diagonal = 20.0
        neighbors = (
            (1, 0, 4.0), (-1, 0, 4.0),
            (0, 1, 4.0), (0, -1, 4.0),
            (1, 1, 1.0), (1, -1, 1.0),
            (-1, 1, 1.0), (-1, -1, 1.0),
        )
        rhs_factor = -6.0 * spacing_squared
    else:
        raise ValueError("stencil must be '5pt' or '9pt'")

    right_hand_side = rhs_factor * source[row_index, column_index]
    matrix_rows, matrix_columns, matrix_values = [], [], []
    for equation_index in range(n_unknowns):
        i = int(row_index[equation_index])
        j = int(column_index[equation_index])
        matrix_rows.append(equation_index)
        matrix_columns.append(equation_index)
        matrix_values.append(diagonal)
        for di, dj, weight in neighbors:
            neighbor_i, neighbor_j = i + di, j + dj
            if not (0 <= neighbor_i < n_grid and 0 <= neighbor_j < n_grid):
                continue
            neighbor_id = unknown_id[neighbor_i, neighbor_j]
            if neighbor_id >= 0:
                matrix_rows.append(equation_index)
                matrix_columns.append(int(neighbor_id))
                matrix_values.append(-weight)
            else:
                right_hand_side[equation_index] += weight * u[neighbor_i, neighbor_j]

    system_matrix = coo_matrix(
        (matrix_values, (matrix_rows, matrix_columns)),
        shape=(n_unknowns, n_unknowns),
    ).tocsr()
    inverse_diagonal = 1.0 / system_matrix.diagonal()
    preconditioner = LinearOperator(
        system_matrix.shape,
        matvec=lambda values: inverse_diagonal * values,
        dtype=system_matrix.dtype,
    )
    try:
        solution, info = cg(
            system_matrix,
            right_hand_side,
            M=preconditioner,
            rtol=cg_rtol,
            atol=0.0,
            maxiter=cg_maxiter,
        )
    except TypeError:  # Support the older SciPy tol argument without changing the linear system.
        solution, info = cg(
            system_matrix,
            right_hand_side,
            M=preconditioner,
            tol=cg_rtol,
            maxiter=cg_maxiter,
        )
    if info != 0:
        print(f"[WARN] F07 CG did not fully converge, info={info}")

    linear_residual = system_matrix @ solution - right_hand_side
    linear_relative_error = np.max(np.abs(linear_residual)) / (
        np.max(np.abs(right_hand_side)) + 1e-12
    )
    u[row_index, column_index] = solution

    core_mask = update_mask[1:-1, 1:-1]
    source_core = source[1:-1, 1:-1]
    u_center = u[1:-1, 1:-1]
    u_north, u_south = u[2:, 1:-1], u[:-2, 1:-1]
    u_east, u_west = u[1:-1, 2:], u[1:-1, :-2]
    if stencil == "5pt":
        laplacian = (
            u_north + u_south + u_east + u_west - 4.0 * u_center
        ) / spacing_squared
    else:
        u_ne, u_nw = u[2:, 2:], u[2:, :-2]
        u_se, u_sw = u[:-2, 2:], u[:-2, :-2]
        laplacian = (
            4.0 * (u_north + u_south + u_east + u_west)
            + u_ne + u_nw + u_se + u_sw
            - 20.0 * u_center
        ) / (6.0 * spacing_squared)
    pde_residual = laplacian - source_core
    pde_relative_error = np.max(np.abs(pde_residual)[core_mask]) / (
        np.max(np.abs(source_core)) + 1e-12
    )
    print(
        f"F07 lin_rel={linear_relative_error:.3e}, "
        f"pde_rel={pde_relative_error:.3e}, stencil={stencil}"
    )

    u[circle_inside] = np.nan
    return x, y, u, source, circle_inside


def _check_output(output: Path, overwrite: bool) -> None:
    if output.exists() and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite reference data {output}; use --out or explicitly pass --overwrite"
        )
    output.parent.mkdir(parents=True, exist_ok=True)


def save_data(x, y, u, source, circle_mask, output=DEFAULT_OUTPUT, overwrite=False):
    output = Path(output)
    _check_output(output, overwrite)
    temporary = output.with_name(f".{output.name}.tmp.npz")
    np.savez_compressed(
        temporary,
        x=x,
        t=y,
        u=u,
        f=source,
        circle_mask=circle_mask,
    )
    os.replace(temporary, output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--n-grid", type=int, default=1000)
    args = parser.parse_args()
    _check_output(args.out, args.overwrite)
    x, y, u, source, circle_mask = solve_perforated_poisson(n_grid=args.n_grid)
    save_data(
        x, y, u, source, circle_mask, args.out, overwrite=args.overwrite
    )


if __name__ == "__main__":
    main()
