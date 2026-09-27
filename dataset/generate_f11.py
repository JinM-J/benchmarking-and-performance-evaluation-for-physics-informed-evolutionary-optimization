"""Generate F11 reference data: Strang splitting and FFT for parametric forced heat flow."""

import argparse
import os
from pathlib import Path

import numpy as np


DEFAULT_OUTPUT = Path(__file__).resolve().parent / "f11.npz"


def solve_parametric_source_heat_equation():
    """Return x,t,mu,u(x,t,mu), input work J(mu), and reference metadata."""
    x_min, x_max, final_time = -2.0, 2.0, 2.0
    n_x, n_t, n_mu = 500, 2001, 1000
    x = np.linspace(x_min, x_max, n_x, endpoint=False)
    t = np.linspace(0.0, final_time, n_t)
    dx = (x_max - x_min) / n_x
    dt = final_time / (n_t - 1)
    diffusion = 0.05
    wave_numbers = 2.0 * np.pi * np.fft.fftfreq(n_x, d=dx)
    diffusion_factor = np.exp(-diffusion * wave_numbers ** 2 * dt)
    mu_values = np.linspace(-1.0, 1.0, n_mu)
    first_time_index, last_time_index = 0, int(2.0 / dt)
    spatial_mask = (x >= x_min) & (x <= x_max)
    spatial_sine = np.sin(np.pi * x[spatial_mask])
    temporal_cosine = np.cos(
        2.0 * np.pi * t[first_time_index:last_time_index + 1]
    )

    def simulate(mu):
        mu = float(mu)
        state = 0.5 * np.sin(np.pi * x)
        trajectory = np.zeros((n_t, n_x))
        trajectory[0] = state
        for n in range(n_t - 1):
            source = (
                (1.0 + mu)
                * 2.0
                * np.sin(np.pi * x)
                * np.cos(2.0 * np.pi * t[n])
            )
            state = state + 0.5 * dt * source
            state = np.real(np.fft.ifft(np.fft.fft(state) * diffusion_factor))
            state = state + 0.5 * dt * source
            trajectory[n + 1] = state
        return trajectory

    dtype = np.float32
    u = np.zeros((n_x, n_t, n_mu), dtype=dtype)
    objective = np.zeros(n_mu, dtype=dtype)
    for index, mu in enumerate(mu_values):
        trajectory = simulate(mu)
        u[:, :, index] = trajectory.T.astype(dtype)
        state_window = trajectory[
            first_time_index:last_time_index + 1, spatial_mask
        ].astype(np.float64)
        source_window = (
            (1.0 + float(mu))
            * 2.0
            * (temporal_cosine[:, None] * spatial_sine[None, :])
        )
        objective[index] = (state_window * source_window).sum() * dx * dt
        if (index + 1) % 200 == 0:
            print(f"F11 [{index + 1}/{n_mu}]")

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
        "u_layout": np.array("x,t,mu"),
        "j_def": np.array(
            "J_in = int_{t1..t2} int_{x1..x2} u(x,t;mu) * s(x,t;mu) dx dt"
        ),
        "s_def": np.array(
            "s(x,t;mu) = (1+mu)*2*sin(pi*x)*cos(2*pi*t)"
        ),
        "time_scheme": np.array("Strang(source half) + FFT diffusion"),
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
    arrays = solve_parametric_source_heat_equation()
    save_data(*arrays, output=args.out, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
