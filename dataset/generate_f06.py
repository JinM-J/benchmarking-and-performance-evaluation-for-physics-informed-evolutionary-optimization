"""Generate F06 reference data: smooth background and a moving local packet.

Manufactured solution:
    u_t + c*u_x - nu*u_xx = s(x,t)
    u = 0.3*exp(-t)*sin(pi*x) + A*exp(-((x-x_s(t))/w)^2)

Existing f06.npz files are protected from overwrite by default."""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np


DEFAULT_OUTPUT = Path(__file__).resolve().parent / "f06.npz"

# Problem constants must match problems/f06.py.
XMIN, XMAX = -1.0, 1.0
TMIN, TMAX = 0.0, 1.0
C_CONV = 0.5
NU = 5e-4
A_PKT = 0.03
W_PKT = 0.10
X_S0 = -0.25
T_STAR = 0.7
ALPHA_T = 0.2
NX, NT = 1024, 401


def background(x, t):
    """Smooth background b(x,t)=0.3*exp(-t)*sin(pi*x)."""
    return 0.3 * np.exp(-t) * np.sin(np.pi * x)


def packet(x, t):
    """Local packet g(x,t)=A*exp(-z^2), z=(x-x_s(t))/w."""
    z = (x - (X_S0 + C_CONV * t)) / W_PKT
    return A_PKT * np.exp(-z ** 2)


def exact_u(x, t):
    return background(x, t) + packet(x, t)


def source_s(x, t):
    """Manufactured source making exact_u satisfy the convection-diffusion PDE."""
    b = background(x, t)
    z = (x - (X_S0 + C_CONV * t)) / W_PKT
    return (
        -b
        + 0.3 * C_CONV * np.pi * np.exp(-t) * np.cos(np.pi * x)
        + NU * np.pi ** 2 * b
        + (NU * A_PKT / W_PKT ** 2)
        * (2 - 4 * z ** 2)
        * np.exp(-z ** 2)
    )


def generate_exact_field(refinement=24):
    """Return x,t,u and a JSON string describing data-generation settings."""
    if not isinstance(refinement, int) or refinement < 1:
        raise ValueError("refinement must be a positive integer")
    x = np.linspace(XMIN, XMAX, (NX - 1) * refinement + 1)
    t = np.linspace(TMIN, TMAX, (NT - 1) * refinement + 1)
    x_grid, t_grid = np.meshgrid(x, t, indexing="ij")
    u = exact_u(x_grid, t_grid)
    if not np.isfinite(u).all():
        raise FloatingPointError("F06 exact field contains nonfinite values")

    # Interior finite-difference residual checks the manufactured solution; it is not saved.
    dt = t[1] - t[0]
    dx = x[1] - x[0]
    source = source_s(x_grid, t_grid)
    u_t = (u[:, 2:] - u[:, :-2]) / (2 * dt)
    u_x = (u[2:, :] - u[:-2, :]) / (2 * dx)
    u_xx = (u[2:, :] - 2 * u[1:-1, :] + u[:-2, :]) / dx ** 2
    residual = (
        u_t[1:-1, :]
        + C_CONV * u_x[:, 1:-1]
        - NU * u_xx[:, 1:-1]
        - source[1:-1, 1:-1]
    )
    print(
        "F06 manufactured-solution numerical residual: "
        f"max={np.abs(residual).max():.2e}, "
        f"mean={np.abs(residual).mean():.2e}"
    )

    x_star = X_S0 + C_CONV * T_STAR
    metadata = {
        "provenance": "manufactured exact solution (no numerical PDE solve error)",
        "pde": "u_t + c*u_x - nu*u_xx = s(x,t)",
        "c": C_CONV,
        "nu": NU,
        "source_formula": "-b + 0.3*c*pi*e^-t*cos(pi*x) + nu*pi^2*b "
        "+ (nu*A/w^2)(2-4z^2)e^-z^2",
        "background": "0.3*e^-t*sin(pi*x)",
        "packet": {"A": A_PKT, "w": W_PKT, "x_s(t)": f"{X_S0}+{C_CONV}*t"},
        "ic_bc": "Boundary values from the exact solution",
        "reference_point": {"x_star": x_star, "t_star": T_STAR},
        "objective": "F = (1-(u-b)/A)^2 + alpha*(t-t*)^2",
        "alpha_t": ALPHA_T,
        "design_intent": "A small local signal on a smooth background distinguishes full-field MSE "
        "from local objective accuracy; omitting the packet has an MSE cost of order 1e-4",
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    mse_ignore = A_PKT ** 2 * W_PKT * np.sqrt(np.pi / 2) / (XMAX - XMIN)
    print(
        f"F06 reference point x*={x_star:.6f}, t*={T_STAR}; "
        f"Global MSE when omitting the packet: approximately {mse_ignore:.2e}"
    )
    return x, t, u, np.array(json.dumps(metadata, ensure_ascii=False))


def _check_output(output: Path, overwrite: bool) -> None:
    if output.exists() and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite reference data {output}; use --out or explicitly pass --overwrite"
        )
    output.parent.mkdir(parents=True, exist_ok=True)


def save_data(x, t, u, meta_json, output=DEFAULT_OUTPUT, overwrite=False):
    output = Path(output)
    _check_output(output, overwrite)
    temporary = output.with_name(f".{output.name}.tmp.npz")
    np.savez(temporary, x=x, t=t, u=u, meta_json=meta_json)
    os.replace(temporary, output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--refinement", type=int, default=24)
    args = parser.parse_args()
    _check_output(args.out, args.overwrite)
    arrays = generate_exact_field(args.refinement)
    save_data(*arrays, output=args.out, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
