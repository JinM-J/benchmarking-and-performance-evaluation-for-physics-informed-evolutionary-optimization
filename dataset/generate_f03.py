"""Generate F03 reaction-diffusion reference data."""

import argparse
import os
from pathlib import Path

import numpy as np


DEFAULT_OUTPUT = Path(__file__).resolve().parent / "f03.npz"


def solve_reaction_diffusion():
    """Solve the Allen-Cahn-type reaction-diffusion equation; return x,t,u(x,t)."""
    alpha = 0.0001
    dx, dt = 5e-4, 5e-5
    n_x, n_t = 20001, 20001
    if alpha * dt / dx ** 2 >= 0.5:
        raise ValueError("F03 explicit scheme violates the diffusion stability condition")

    x = np.linspace(-5.0, 5.0, n_x)
    t = np.linspace(0.0, 1.0, n_t)
    u = np.zeros((n_x, n_t))
    u[:, 0] = -(2.15 / np.pi) * x * np.cos(np.pi * x)
    u[0, :] = u[-1, :]
    u[1, :] = u[-2, :]

    for n in range(n_t - 1):
        u_n = u[:, n]
        u_xx = (u_n[2:] - 2 * u_n[1:-1] + u_n[:-2]) / dx ** 2
        u[1:-1, n + 1] = u_n[1:-1] + dt * (
            alpha * u_xx - 5 * u_n[1:-1] ** 3 + 5 * u_n[1:-1]
        )
        u[0, n + 1] = u[-1, n + 1]
        u[-1, n + 1] = u[0, n + 1]
        u[1, n + 1] = u[-2, n + 1]
        u[-2, n + 1] = u[1, n + 1]
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
    np.savez(temporary, x=x, t=t, u=u)
    os.replace(temporary, output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    _check_output(args.out, args.overwrite)
    x, t, u = solve_reaction_diffusion()
    save_data(x, t, u, args.out, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
