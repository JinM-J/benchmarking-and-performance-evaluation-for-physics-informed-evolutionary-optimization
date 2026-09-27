"""Generate F09 parametric reaction-diffusion data using Strang splitting and FFT."""

import argparse
import os
from pathlib import Path

import numpy as np


DEFAULT_OUTPUT = Path(__file__).resolve().parent / "f09.npz"


def solve_parametric_reaction_diffusion():
    """Return x,t,mu,u(x,t,mu),J(mu), and reference metadata."""
    x_min, x_max, final_time = -2.0, 2.0, 2.0
    n_x, n_t, n_mu = 500, 2001, 1000
    x = np.linspace(x_min, x_max, n_x, endpoint=False)
    t = np.linspace(0.0, final_time, n_t)
    dx = (x_max - x_min) / n_x
    dt = final_time / (n_t - 1)
    diffusion, alpha = 0.05, 1.0
    wave_numbers = 2.0 * np.pi * np.fft.fftfreq(n_x, d=dx)
    diffusion_factor = np.exp(-diffusion * wave_numbers ** 2 * dt)
    mu_values = np.linspace(-1.0, 1.0, n_mu)
    first_time_index, last_time_index = 0, int(2.0 / dt)
    spatial_mask = (x >= x_min) & (x <= x_max)

    def simulate(mu):
        state = (
            np.sin(np.pi * (x + mu))
            + mu * np.sin(np.pi * 3.0 * mu * (x - mu))
        )
        trajectory = np.zeros((n_t, n_x))
        trajectory[0] = state
        for n in range(n_t - 1):
            state = state + 0.5 * dt * alpha * state ** 2 * (1.0 - state)
            state = np.real(np.fft.ifft(np.fft.fft(state) * diffusion_factor))
            state = state + 0.5 * dt * alpha * state ** 2 * (1.0 - state)
            trajectory[n + 1] = state
        return trajectory

    dtype = np.float32
    u = np.zeros((n_x, n_t, n_mu), dtype=dtype)
    objective = np.zeros(n_mu, dtype=dtype)
    for index, mu in enumerate(mu_values):
        trajectory = simulate(mu)
        u[:, :, index] = trajectory.T.astype(dtype)
        objective[index] = (
            trajectory[first_time_index:last_time_index + 1, spatial_mask].sum()
            * dx
            * dt
        )
        if (index + 1) % 200 == 0:
            print(f"F09 [{index + 1}/{n_mu}]")

    metadata = {
        "Nx": np.int32(n_x),
        "Nt": np.int32(n_t),
        "N_MU": np.int32(n_mu),
        "n1": np.int32(first_time_index),
        "n2": np.int32(last_time_index),
        "x_a": np.float32(x_min),
        "x_b": np.float32(x_max),
        "T": np.float32(final_time),
        "dx": np.float32(dx),
        "dt": np.float32(dt),
        "MU_MIN": np.float32(-1.0),
        "MU_MAX": np.float32(1.0),
        "x1": np.float32(x_min),
        "x2": np.float32(x_max),
        "t1": np.float32(0.0),
        "t2": np.float32(final_time),
        "D": np.float32(diffusion),
        "alpha": np.float32(alpha),
        "u_layout": np.array("x,t,mu"),
        "time_scheme": np.array("Strang(reaction half) + FFT diffusion"),
    }
    return (
        x.astype(dtype),
        t.astype(dtype),
        mu_values.astype(dtype),
        u,
        objective,
        metadata,
    )


def _check_output(output: Path, overwrite: bool) -> None:
    if output.exists() and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite reference data {output}; use --out or explicitly pass --overwrite"
        )
    output.parent.mkdir(parents=True, exist_ok=True)


def save_data(x, t, mu, u, objective, metadata, output=DEFAULT_OUTPUT, overwrite=False):
    output = Path(output)
    _check_output(output, overwrite)
    temporary = output.with_name(f".{output.name}.tmp.npz")
    np.savez(
        temporary,
        x=x,
        t=t,
        mu=mu,
        u=u,
        j=objective,
        **metadata,
    )
    os.replace(temporary, output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    _check_output(args.out, args.overwrite)
    arrays = solve_parametric_reaction_diffusion()
    save_data(*arrays, output=args.out, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
