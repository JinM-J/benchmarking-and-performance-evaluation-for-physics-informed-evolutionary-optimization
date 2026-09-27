"""Generate F08 Kuramoto-Sivashinsky reference data."""

import argparse
import os
from pathlib import Path

import numpy as np


DEFAULT_OUTPUT = Path(__file__).resolve().parent / "f08.npz"


def solve_kuramoto_sivashinsky():
    """Generate F08 states using periodic finite differences and RK4."""
    length, final_time = 2.0, 1.0
    n_x = 200
    alpha = 100 / 16
    beta = 100 / 16 ** 2
    gamma = 100 / 16 ** 4
    dx = length / n_x
    dt = 1e-6
    store_every = 100
    n_stored = int(round(final_time / (dt * store_every)))

    x = np.linspace(0.0, length, n_x, endpoint=False)
    t = np.linspace(0.0, final_time, n_stored + 1)
    if gamma * 16 / dx ** 4 * dt >= 2.785:
        raise ValueError("F08 RK4 violates the fourth-order stiffness stability condition")

    state = np.cos(x) * (1 + np.sin(x))
    u = np.zeros((n_x, n_stored + 1))
    u[:, 0] = state

    def laplacian(values):
        return (
            np.roll(values, -1) - 2 * values + np.roll(values, 1)
        ) / dx ** 2

    def bi_laplacian(values):
        return (
            np.roll(values, -2)
            - 4 * np.roll(values, -1)
            + 6 * values
            - 4 * np.roll(values, 1)
            + np.roll(values, 2)
        ) / dx ** 4

    def right_hand_side(values):
        first_derivative = (
            np.roll(values, -1) - np.roll(values, 1)
        ) / (2 * dx)
        return (
            -alpha * values * first_derivative
            - beta * laplacian(values)
            - gamma * bi_laplacian(values)
        )

    stored_index = 0
    for step in range(n_stored * store_every):
        k1 = right_hand_side(state)
        k2 = right_hand_side(state + 0.5 * dt * k1)
        k3 = right_hand_side(state + 0.5 * dt * k2)
        k4 = right_hand_side(state + dt * k3)
        state = state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        if (step + 1) % store_every == 0:
            stored_index += 1
            u[:, stored_index] = state
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
    x, t, u = solve_kuramoto_sivashinsky()
    save_data(x, t, u, args.out, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
