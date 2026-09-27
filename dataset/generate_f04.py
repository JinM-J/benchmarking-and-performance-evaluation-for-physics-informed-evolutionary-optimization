"""Generate F04 reference data with 1200 spatial intervals and 640000 time steps."""

import argparse
import os
from pathlib import Path

import numpy as np


DEFAULT_OUTPUT = Path(__file__).resolve().parent / "f04.npz"


def solve_source_heat_equation(n_x_intervals=1200, n_time_steps=640000):
    """Solve the heat equation with a piecewise spatial source; return x,t,u."""
    length, final_time, beta_u = np.pi, 2.0, 1.5
    if not isinstance(n_x_intervals, int) or n_x_intervals < 4 or n_x_intervals % 4:
        raise ValueError("The spatial interval count must be a positive multiple of 4 to align source segments")
    if not isinstance(n_time_steps, int) or n_time_steps < 1:
        raise ValueError("The time-step count must be a positive integer")
    dt = final_time / n_time_steps
    dx = length / n_x_intervals
    if dt * (2.0 / dx ** 2 + beta_u) >= 1.0:
        raise ValueError("F04 explicit scheme violates the diffusion stability condition")

    x = np.linspace(0.0, length, n_x_intervals + 1)
    # Align source interfaces with grid nodes to avoid roundoff selecting the left interval.
    x[n_x_intervals // 4] = length / 4
    x[n_x_intervals // 2] = length / 2
    x[3 * n_x_intervals // 4] = 3 * length / 4
    t = np.linspace(0.0, final_time, n_time_steps + 1)
    temperature = np.zeros((n_x_intervals + 1, n_time_steps + 1))

    interior_x = x[1:-1]
    spatial_basis = np.zeros((4, interior_x.size))
    spatial_basis[0, (0 <= interior_x) & (interior_x < np.pi / 4)] = 1
    spatial_basis[1, (np.pi / 4 <= interior_x) & (interior_x < np.pi / 2)] = 1
    spatial_basis[2, (np.pi / 2 <= interior_x) & (interior_x < 3 * np.pi / 4)] = 1
    spatial_basis[3, (3 * np.pi / 4 <= interior_x) & (interior_x <= np.pi)] = 1

    for n in range(n_time_steps):
        temporal_source = np.array(
            [1.1 + 5 * np.sin(t[n] / 4 + i / 10) for i in range(1, 5)]
        )
        source_term = spatial_basis.T @ temporal_source
        temperature_n = temperature[:, n]
        diffusion = (
            temperature_n[:-2]
            - 2 * temperature_n[1:-1]
            + temperature_n[2:]
        ) / dx ** 2
        temperature[1:-1, n + 1] = temperature_n[1:-1] + dt * (
            diffusion + beta_u * (source_term - temperature_n[1:-1])
        )
    return x, t, temperature


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
    parser.add_argument("--n-x-intervals", type=int, default=1200)
    parser.add_argument("--n-time-steps", type=int, default=640000)
    args = parser.parse_args()
    _check_output(args.out, args.overwrite)
    x, t, u = solve_source_heat_equation(args.n_x_intervals, args.n_time_steps)
    save_data(x, t, u, args.out, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
