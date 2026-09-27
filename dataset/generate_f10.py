"""Generate F10 parametric Burgers data on a doubled solver grid."""

import argparse
import os
from pathlib import Path

import numpy as np


DEFAULT_OUTPUT = Path(__file__).resolve().parent / "f10.npz"


def solve_parametric_burgers():
    """Return x,t,nu,lognu,u(x,t,nu),J(nu), and reference metadata."""
    from scipy.fft import dst, idst

    x_min, x_max, final_time = -1.0, 1.0, 1.0
    n_x, n_t_solve, n_nu = 1000, 4001, 1000
    x = np.linspace(x_min, x_max, n_x, endpoint=True)
    t_solve = np.linspace(0.0, final_time, n_t_solve)
    dx = (x_max - x_min) / (n_x - 1)
    dt_solve = final_time / (n_t_solve - 1)
    nu_values = np.logspace(np.log10(1.0), np.log10(1e-3), n_nu)
    lognu_values = np.log(nu_values).astype(np.float32)

    def advection_right_hand_side(state):
        state = state.copy()
        state[0] = state[-1] = 0.0
        flux = 0.5 * state * state
        state_left, state_right = state[:-1], state[1:]
        wave_speed = np.maximum(np.abs(state_left), np.abs(state_right))
        interface_flux = (
            0.5 * (flux[:-1] + flux[1:])
            - 0.5 * wave_speed * (state_right - state_left)
        )
        rhs = np.zeros_like(state)
        rhs[1:-1] = -(interface_flux[1:] - interface_flux[:-1]) / dx
        return rhs

    n_interior = n_x - 2
    modes = np.arange(1, n_interior + 1, dtype=np.float64)
    eigenvalues = (
        4.0
        * np.sin(np.pi * modes / (2.0 * (n_interior + 1))) ** 2
        / dx ** 2
    )

    def diffuse(state, viscosity):
        transformed = dst(state[1:-1], type=1, norm="ortho")
        transformed *= np.exp(-viscosity * eigenvalues * dt_solve)
        state = state.copy()
        state[1:-1] = idst(transformed, type=1, norm="ortho")
        state[0] = state[-1] = 0.0
        return state

    def simulate(viscosity):
        viscosity = float(viscosity)
        state = -np.sin(np.pi * x)
        state[0] = state[-1] = 0.0
        trajectory = np.zeros((n_t_solve, n_x))
        trajectory[0] = state
        for n in range(n_t_solve - 1):
            k1 = advection_right_hand_side(state)
            predictor = state + dt_solve * k1
            predictor[0] = predictor[-1] = 0.0
            k2 = advection_right_hand_side(predictor)
            state = diffuse(
                state + 0.5 * dt_solve * (k1 + k2), viscosity
            )
            trajectory[n + 1] = state
        return trajectory

    dtype = np.float32
    t = t_solve[::2]
    u = np.zeros((n_x, t.size, n_nu), dtype=dtype)
    objective = np.zeros(n_nu, dtype=dtype)
    for index, viscosity in enumerate(nu_values):
        trajectory = simulate(viscosity)
        u[:, :, index] = trajectory[::2].T.astype(dtype)
        objective[index] = (trajectory ** 2).sum() * dx * dt_solve
        if (index + 1) % 100 == 0:
            print(f"F10 [{index + 1}/{n_nu}]")

    metadata = {
        "Nx": np.int32(n_x),
        "Nt": np.int32(t.size),
        "N_NU": np.int32(n_nu),
        "n1": np.int32(0),
        "n2": np.int32(t.size - 1),
        "x_a": np.float32(x_min),
        "x_b": np.float32(x_max),
        "T": np.float32(final_time),
        "dx": np.float32(dx),
        "dt": np.float32(final_time / (t.size - 1)),
        "NU_MIN": np.float32(1e-3),
        "NU_MAX": np.float32(1.0),
        "x1": np.float32(x_min),
        "x2": np.float32(x_max),
        "t1": np.float32(0.0),
        "t2": np.float32(final_time),
        "BC_LEFT": np.float32(0.0),
        "BC_RIGHT": np.float32(0.0),
        "burgers_form": np.array("B"),
        "bc_type": np.array("D"),
        "time_scheme": np.array(
            "RK2+Rusanov + diffusion(DST); solved at 2x grid"
        ),
        "J_kind": np.array("J2_full"),
        "u_layout": np.array("x,t,nu"),
        "have_scipy_dst": np.array(True),
    }
    return (
        x.astype(dtype),
        t.astype(dtype),
        nu_values.astype(dtype),
        lognu_values,
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


def save_data(
    x,
    t,
    nu,
    lognu,
    u,
    objective,
    metadata,
    output=DEFAULT_OUTPUT,
    overwrite=False,
):
    output = Path(output)
    _check_output(output, overwrite)
    temporary = output.with_name(f".{output.name}.tmp.npz")
    np.savez(
        temporary,
        x=x,
        t=t,
        nu=nu,
        lognu=lognu,
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
    arrays = solve_parametric_burgers()
    save_data(*arrays, output=args.out, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
