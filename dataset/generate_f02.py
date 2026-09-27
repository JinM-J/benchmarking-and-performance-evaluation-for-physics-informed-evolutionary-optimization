"""Generate F02 reference data on the refined 16001x6401 grid with CFL=1."""

import argparse
import os
from pathlib import Path

import numpy as np


DEFAULT_OUTPUT = Path(__file__).resolve().parent / "f02.npz"


def solve_wave_equation(refinement=16):
    """Solve the wave equation with zero initial velocity; return x,t,u(x,t)."""
    c = 2.0
    if not isinstance(refinement, int) or refinement < 1:
        raise ValueError("refinement must be a positive integer")
    dx = 0.005 / refinement
    dt = dx / c  # At CFL=1, leapfrog is pointwise exact for this one-dimensional wave discretization.
    n_x, n_t = 1000 * refinement + 1, 400 * refinement + 1
    x = np.linspace(0.0, 5.0, n_x)
    t = np.linspace(0.0, 1.0, n_t)
    u = np.zeros((n_x, n_t))
    u[:, 0] = -2.5 * np.sin(np.pi * x) + 2.6 * np.sin(4 * np.pi * x)

    # Zero initial velocity leaves only even spatial derivatives in the Taylor start.
    u_xx0 = (
        2.5 * np.pi ** 2 * np.sin(np.pi * x)
        - 2.6 * (4 * np.pi) ** 2 * np.sin(4 * np.pi * x)
    )
    u_xxxx0 = (
        -2.5 * np.pi ** 4 * np.sin(np.pi * x)
        + 2.6 * (4 * np.pi) ** 4 * np.sin(4 * np.pi * x)
    )
    u[:, 1] = (
        u[:, 0]
        + 0.5 * (c * dt) ** 2 * u_xx0
        + (c * dt) ** 4 / 24.0 * u_xxxx0
    )
    u[0, :] = 0.0
    u[-1, :] = 0.0

    for n in range(1, n_t - 1):
        u[1:-1, n + 1] = (
            2 * u[1:-1, n]
            - u[1:-1, n - 1]
            + u[2:, n]
            - 2 * u[1:-1, n]
            + u[:-2, n]
        )

    def initial_profile(y):
        return -2.5 * np.sin(np.pi * y) + 2.6 * np.sin(4 * np.pi * y)

    def odd_periodic_extension(y):
        y_mod = np.mod(y, 10.0)
        return np.where(
            y_mod <= 5.0,
            initial_profile(y_mod),
            -initial_profile(10.0 - y_mod),
        )

    x_grid, t_grid = np.meshgrid(x, t, indexing="ij")
    u_exact = 0.5 * (
        odd_periodic_extension(x_grid - c * t_grid)
        + odd_periodic_extension(x_grid + c * t_grid)
    )
    relative_error = np.max(np.abs(u - u_exact)) / np.max(np.abs(u_exact))
    print(f"F02 d'Alembert self-check maximum relative error: {relative_error:.3e}")
    return x, t, u


def _check_output(output: Path, overwrite: bool) -> None:
    if output.exists() and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite reference data {output}; use --out or explicitly pass --overwrite"
        )
    output.parent.mkdir(parents=True, exist_ok=True)


def save_data(x, t, u, output=DEFAULT_OUTPUT, overwrite=False):
    output = Path(output)
    _check_output(output, overwrite)
    temporary = output.with_name(f".{output.name}.tmp.npz")
    np.savez_compressed(temporary, x=x, t=t, u=u)
    os.replace(temporary, output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--refinement", type=int, default=16)
    args = parser.parse_args()
    _check_output(args.out, args.overwrite)
    x, t, u = solve_wave_equation(args.refinement)
    save_data(x, t, u, args.out, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
