"""Generate F01 reference data using the refined Burgers solver.

Replacing f01.npz requires --overwrite. Use --out to direct
validation runs to a separate file."""

import argparse
import os
from pathlib import Path

import numpy as np


DEFAULT_OUTPUT = Path(__file__).resolve().parent / "f01.npz"


def solve_burgers():
    """Solve u_t + u*u_x = nu*u_xx and return x,t,u(x,t)."""
    nu = 0.01 / np.pi
    dx, dt = 2.0 / 6000, 1.25e-5
    n_x, n_t = 6001, 80001
    if nu * dt / dx ** 2 >= 0.5:
        raise ValueError("F01 explicit scheme violates the diffusion stability condition")

    x = np.linspace(-1.0, 1.0, n_x)
    t = np.linspace(0.0, 1.0, n_t)
    u = np.zeros((n_x, n_t))
    u[:, 0] = -np.sin(np.pi * x)

    for n in range(n_t - 1):
        u_n = u[:, n]
        u_x = (u_n[2:] - u_n[:-2]) / (2 * dx)
        u_xx = (u_n[2:] - 2 * u_n[1:-1] + u_n[:-2]) / dx ** 2
        u[1:-1, n + 1] = u_n[1:-1] + dt * (
            -u_n[1:-1] * u_x + nu * u_xx
        )
    return x, t, u


def _check_output(output: Path, overwrite: bool) -> None:
    if output.exists() and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite reference data {output}; use --out or explicitly pass --overwrite"
        )
    output.parent.mkdir(parents=True, exist_ok=True)


def save_data(x, t, u, output=DEFAULT_OUTPUT, overwrite=False):
    """Atomically write F01 arrays with fixed keys x,t,u."""
    output = Path(output)
    _check_output(output, overwrite)
    temporary = output.with_name(f".{output.name}.tmp.npz")
    np.savez(temporary, x=x, t=t, u=u)
    os.replace(temporary, output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    _check_output(args.out, args.overwrite)
    x, t, u = solve_burgers()
    save_data(x, t, u, args.out, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
