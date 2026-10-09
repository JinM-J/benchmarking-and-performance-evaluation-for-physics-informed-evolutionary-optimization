"""F11 discrete reference optimum and independent continuous-integral verification.

The benchmark optimization problem uses a fixed 10x10 quadrature on an
interpolated field. --data-dir solves that discrete reference problem.
--continuous instead checks the distinct continuous PDE/integral problem,
for which J=0.5 is unreachable on mu in [-1,1], without loading a dataset.
Neither mode generates PDE data or trains a model.
"""
import argparse
import ast
from fractions import Fraction
import json
from pathlib import Path
import sys

import mpmath as mp
import numpy as np
from scipy.integrate import quad
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from problems.f11 import F11


def amplitude(t, mu, diffusion):
    """Solve A'+lambda*A=2*(1+mu)*cos(omega*t), A(0)=1/2."""
    lam, omega = diffusion * np.pi**2, 2 * np.pi
    decay = np.exp(-lam * np.asarray(t))
    return (0.5 * decay + 2 * (1 + np.asarray(mu))
            * (lam * np.cos(omega * t) + omega * np.sin(omega * t)
               - lam * decay) / (lam * lam + omega * omega))


def exact(queries, diffusion):
    x, t, mu = np.asarray(queries, dtype=float).T
    return np.sin(np.pi * x) * amplitude(t, mu, diffusion)


def historical_exact(path):
    """Load only exact() from a trusted comparison script."""
    source = path.read_text(encoding="utf-8")
    functions = [node for node in ast.parse(source).body
                 if isinstance(node, ast.FunctionDef) and node.name == "exact"]
    if len(functions) != 1:
        raise ValueError("Comparison source must contain exactly one top-level exact()")
    namespace = {"np": np}
    module = ast.Module(body=functions, type_ignores=[])
    exec(compile(module, str(path), "exec"), namespace)
    return namespace["exact"], {
        "filename": path.name,
        "execution_scope": "Only exact() is evaluated; no experiment driver or dataset access.",
    }


def check_physics(diffusion, old_exact=None):
    problem = F11(D=diffusion)
    rng = np.random.default_rng(20260924)
    queries = rng.uniform(problem.query_bounds[:, 0], problem.query_bounds[:, 1], (2048, 3))
    q = torch.tensor(queries, dtype=torch.float64, requires_grad=True)
    x, t, mu = q[:, 0:1], q[:, 1:2], q[:, 2:3]
    lam, omega = diffusion * np.pi**2, 2 * np.pi
    decay = torch.exp(-lam * t)
    state = torch.sin(np.pi * x) * (
        0.5 * decay + 2 * (1 + mu)
        * (lam * torch.cos(omega * t) + omega * torch.sin(omega * t) - lam * decay)
        / (lam * lam + omega * omega))
    gradient = torch.autograd.grad(state.sum(), q, create_graph=True)[0]
    dxx = torch.autograd.grad(gradient[:, 0].sum(), q)[0][:, 0:1]
    residual = problem.physics.residual(
        q, state, {("t",): gradient[:, 1:2], ("x", "x"): dxx})
    errors = {
        "official_pde_residual_max_abs": float(residual.detach().abs().max()),
        "numpy_vs_torch_solution_max_abs": float(np.max(
            np.abs(state.detach().numpy().ravel() - exact(queries, diffusion)))),
    }
    source = (2 * (1 + queries[:, 2]) * np.sin(np.pi * queries[:, 0])
              * np.cos(2 * np.pi * queries[:, 1]))
    errors["official_source_max_abs_error"] = float(np.max(
        np.abs(problem.physics.linear_operator().rhs(queries) - source)))
    initial_conditions = problem.physics.initial_conditions()
    if len(initial_conditions) != 1 or initial_conditions[0].derivative != ():
        raise AssertionError("Unexpected F11 initial-condition interface")
    ic = initial_conditions[0]
    qi = ic.region.sample(512)
    errors["official_ic_max_abs_error"] = float(np.max(
        np.abs(exact(qi, diffusion) - ic.target(qi))))
    boundaries = problem.physics.boundary_conditions()
    if len(boundaries) != 1 or boundaries[0].derivative != ("x",):
        raise AssertionError("Unexpected F11 periodic-derivative interface")
    left, right = [region.sample(512) for region in boundaries[0].region_pair]
    np.testing.assert_array_equal(left[:, 1:], right[:, 1:])
    errors["periodic_value_max_abs_error"] = float(np.max(
        np.abs(exact(left, diffusion) - exact(right, diffusion))))
    dx_left = np.pi * np.cos(np.pi * left[:, 0]) * amplitude(left[:, 1], left[:, 2], diffusion)
    dx_right = np.pi * np.cos(np.pi * right[:, 0]) * amplitude(right[:, 1], right[:, 2], diffusion)
    errors["periodic_dx_max_abs_error"] = float(np.max(np.abs(dx_left - dx_right)))
    if old_exact is not None:
        errors["historical_exact_max_abs_difference"] = float(np.max(
            np.abs(old_exact(queries, diffusion) - exact(queries, diffusion))))
    if not all(np.isfinite(v) and v < 1e-12 for v in errors.values()):
        raise AssertionError(errors)
    return {"status": "PASS", "random_seed": 20260924, "interior_points": 2048,
            "ic_points": 512, "boundary_pairs": 512, "absolute_tolerance": 1e-12,
            "dtype": "float64", **errors}


def check_integral(diffusion_fraction):
    diffusion = mp.mpf(diffusion_fraction.numerator) / diffusion_fraction.denominator
    lam, omega, duration = diffusion * mp.pi**2, 2 * mp.pi, mp.mpf(2)
    denominator = lam * lam + omega * omega
    decay_loss = -mp.expm1(-lam * duration)
    c1 = 2 * lam * decay_loss / denominator
    c2 = (4 * lam * duration / denominator
          - 8 * lam * lam * decay_loss / denominator**2)
    # 1-exp(-lambda*T) < lambda*T and omega>lambda imply c2>0.
    assert 0 < lam < omega and c1 > 0 and c2 > 0
    exact_maximum = 2 * c1 + 4 * c2
    # For T=2, b<=2: J <= 36*lambda/(lambda^2+omega^2) < 9*D.
    # The last comparison with 1/2 uses exact rational arithmetic, not quadrature.
    rational_upper = 9 * diffusion_fraction
    assert rational_upper < Fraction(1, 2)
    rows = []
    space_integral, space_error = quad(lambda x: np.sin(np.pi*x)**2, -2, 2,
                                       epsabs=1e-13, epsrel=1e-13)
    for mu in (-1, -0.5, 0, 0.5, 1):
        b = mp.mpf(str(1 + mu))
        closed = c1 * b + c2 * b * b

        def integrand(t):
            decay = mp.exp(-lam * t)
            amp = (mp.mpf("0.5") * decay + 2 * b
                   * (lam * mp.cos(omega*t) + omega * mp.sin(omega*t) - lam*decay)
                   / denominator)
            return 4 * b * amp * mp.cos(omega*t)

        high_precision = mp.quad(integrand, [mp.mpf(i)/4 for i in range(9)])
        temporal, quad_error = quad(
            lambda t: float(amplitude(t, mu, float(diffusion))) * np.cos(2*np.pi*t),
            0, 2, epsabs=1e-13, epsrel=1e-13)
        scipy_value = 2 * float(b) * space_integral * temporal
        mp_error = abs(high_precision - closed)
        scipy_error = abs(scipy_value - float(closed))
        assert mp_error < mp.mpf("1e-65") and scipy_error < 1e-12
        rows.append({"mu": mu, "closed_form": mp.nstr(closed, 50),
                     "mpmath_quad": mp.nstr(high_precision, 50),
                     "mpmath_abs_difference": mp.nstr(mp_error, 12),
                     "scipy_quad": scipy_value, "scipy_abs_difference": scipy_error,
                     "scipy_temporal_quad_error_estimate": quad_error})
    return {
        "status": "PASS", "mpmath_working_digits": mp.mp.dps,
        "coefficient_b": mp.nstr(c1, 50), "coefficient_b_squared": mp.nstr(c2, 50),
        "monotone_in_b_on_0_2": True,
        "range_mu": [-1, 1], "range_b": [0, 2], "minimum": "0", "minimum_mu": -1,
        "maximum": mp.nstr(exact_maximum, 50), "maximum_mu": 1,
        "target_J": "0.5", "gap_to_target": mp.nstr(mp.mpf("0.5")-exact_maximum, 50),
        "strict_uniform_upper_bound_rational": str(rational_upper),
        "strict_uniform_upper_bound_decimal": float(rational_upper),
        "upper_bound_below_target_checked_with_exact_rationals": True,
        "target_unreachable_for_continuous_integral": True,
        "spatial_sin_squared_integral": space_integral,
        "spatial_quad_error_estimate": space_error,
        "quadrature_cross_checks": rows,
    }


def verify_continuous(historical_source=None):
    torch.set_num_threads(1)
    mp.mp.dps = 80
    old_exact, historical_record = (None, None)
    if historical_source is not None:
        old_exact, historical_record = historical_exact(historical_source)
    cases = []
    for name, fraction in (
        ("nominal_D_0.05", Fraction(1, 20)),
        ("float32_metadata_D", Fraction(float(np.float32(0.05)))),
    ):
        cases.append({"case": name, "D_exact_rational": str(fraction), "D_float64": float(fraction),
                      "physics_checks": check_physics(float(fraction), old_exact),
                      "continuous_integral": check_integral(fraction)})
    result = {
        "status": "PASS", "paper_problem": "F11", "code_problem": "f11",
        "scope": "Continuous PDE solution and true input-work integral; not the official discrete objective.",
        "dataset_loaded": False, "new_pde_solves": 0, "new_optimizations": 0,
        "formula": {
            "definitions": "b=1+mu, lambda=D*pi^2, omega=2*pi, K=lambda^2+omega^2, T=2",
            "solution": "u=sin(pi*x)*[exp(-lambda*t)/2+2*b*(lambda*cos(omega*t)+omega*sin(omega*t)-lambda*exp(-lambda*t))/K]",
            "ode": "A'+lambda*A=2*b*cos(omega*t), A(0)=1/2",
            "integral": "J=int_0^2 int_-2^2 u*s dx dt = 4*b*int_0^2 A(t)*cos(omega*t) dt = c1*b+c2*b^2",
            "c1": "2*lambda*(1-exp(-lambda*T))/K",
            "c2": "4*lambda*T/K-8*lambda^2*(1-exp(-lambda*T))/K^2",
            "monotonicity": "c1>0; 1-exp(-lambda*T)<lambda*T and omega>lambda imply c2>0. Thus dJ/db=c1+2*c2*b>0 on [0,2].",
            "strict_bound": "Drop the negative term and use 1-exp(-lambda*T)<1, b<=2: J<(4+16*T)*lambda/K=36*lambda/K<9*D<1/2. The final comparison is exact rational arithmetic for both D values.",
        },
        "historical_formula_comparison": historical_record,
        "cases": cases,
        "evidence_limits": [
            "The benchmark defines J by 10x10 quadrature on its stored interpolated field. This discrete integral may attain J=0.5.",
            "This continuous-integral obstruction does not invalidate a separately verified optimum of the official discrete objective.",
            "Because the continuous Ackley lower bound zero requires both U=0 and J=0.5, zero is unattainable for this continuous-integral problem. Its actual positive optimum is not computed here.",
            "High-precision quadrature checks support the closed expression; the strict bound below 0.5 additionally follows from symbolic inequalities and exact rational D comparisons.",
            "The metadata case reproduces float32(0.05); no NPZ is read or independently identified by this script.",
        ],
    }
    return result


def solve(data_dir):
    from reference.common import integral
    result = integral.solve("F11", Path(data_dir))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--data-dir", type=Path, help="Solve the discrete reference problem using f11.npz")
    mode.add_argument("--continuous", action="store_true", help="Check the separate continuous integral without loading data")
    parser.add_argument("--historical-source", type=Path, help="Optional independent exact-solution source for --continuous")
    parser.add_argument("--output", type=Path, help="New JSON file; defaults to stdout")
    args = parser.parse_args()
    if args.output is not None and args.output.exists():
        parser.error(f"Output already exists: {args.output}")
    if args.historical_source is not None and not args.continuous:
        parser.error("--historical-source requires --continuous")
    result = verify_continuous(args.historical_source) if args.continuous else solve(args.data_dir)
    payload = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output is None:
        print(payload, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(payload)
        print(f"PASS: F11 reference verification; {args.output}")


if __name__ == "__main__":
    main()
