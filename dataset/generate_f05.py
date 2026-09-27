"""Generate F05 forced heat-equation reference data with periodic boundaries."""

import argparse
import os
from pathlib import Path

import numpy as np


DEFAULT_OUTPUT = Path(__file__).resolve().parent / "f05.npz"


def solve_periodic_source_heat_equation():
    """Solve F05 using the reference discretization; return x,t,u(x,t)."""
    length = 3.0
    final_time = 2.0
    n_x_intervals = 1000
    n_time_steps = 50000
    dx = length / n_x_intervals
    dt = final_time / n_time_steps
    alpha = 0.01

    x = np.linspace(0.0, length, n_x_intervals + 1)
    t = np.linspace(0.0, final_time, n_time_steps + 1)
    u = np.zeros((n_x_intervals + 1, n_time_steps + 1))
    u[:, 0] = 2.1 * x * np.sin(np.pi * x)

    def heat_source(x_value, t_value):
        return 2 * np.sin(np.pi * x_value) * np.cos(2 * np.pi * t_value)

    for n in range(n_time_steps):
        for i in range(1, n_x_intervals):
            diffusion = (u[i - 1, n] - 2 * u[i, n] + u[i + 1, n]) / dx ** 2
            source = heat_source(x[i], t[n])
            u[i, n + 1] = u[i, n] + dt * (alpha * diffusion + source)
        u[0, n + 1] = u[-1, n + 1]
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
    args = parser.parse_args()
    _check_output(args.out, args.overwrite)
    x, t, u = solve_periodic_source_heat_equation()
    save_data(x, t, u, args.out, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
