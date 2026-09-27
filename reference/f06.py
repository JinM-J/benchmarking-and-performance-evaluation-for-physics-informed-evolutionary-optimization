#!/usr/bin/env python3
"""Verify the analytic solution and continuous optimum of F6.

From the repository root: python reference/f06.py --output report.json
Optionally --dataset /path/to/f06.npz checks existing metadata and the
objective API; only tiny metadata and coordinate members are read, never u.
"""
from pathlib import Path
import argparse
import importlib.util
import json
import sys

sys.dont_write_bytecode = True


def verify(root, dataset=None, n_points=2048, seed=20260924):
    import numpy as np
    import torch
    root = Path(root).resolve()
    sys.path.insert(0, str(root))
    import problems.f06 as problem_module
    spec = importlib.util.spec_from_file_location("f6_generation_definition", root / "dataset/generate_f06.py")
    generator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generator)
    expected = dict(C_CONV=.5, NU=.0005, A_PKT=.03, W_PKT=.10, X_S0=-.25, T_STAR=.7, ALPHA_T=.2)
    for key, value in expected.items():
        if getattr(problem_module, key) != value or getattr(generator, key) != value:
            raise AssertionError(f"Definition drift: {key}")
    problem = problem_module.F06()
    np.testing.assert_array_equal(problem.decision_bounds, [[-1., 1.], [0., 1.]])
    np.testing.assert_array_equal([generator.XMIN, generator.XMAX, generator.TMIN, generator.TMAX], [-1., 1., 0., 1.])
    rng = np.random.default_rng(seed)
    q = rng.uniform([-1., 0.], [1., 1.], (n_points, 2))
    qt = torch.tensor(q, dtype=torch.float64, requires_grad=True)
    x, t = qt[:, 0:1], qt[:, 1:2]
    # Independently written fixed expression, avoiding the code's source formula.
    u = .3 * torch.exp(-t) * torch.sin(torch.pi*x) + .03 * torch.exp(-((x+.25-.5*t)/.10)**2)
    gradient = torch.autograd.grad(u.sum(), qt, create_graph=True)[0]
    dxx = torch.autograd.grad(gradient[:, 0].sum(), qt)[0][:, 0:1]
    residual = problem.physics.residual(qt, u, {("t",): gradient[:, 1:2], ("x",): gradient[:, 0:1], ("x", "x"): dxx})
    checks = dict(
        pde_autodiff_max_abs=float(residual.detach().abs().max()),
        source_generator_max_abs=float(np.max(abs(problem_module._source_np(q)-generator.source_s(q[:, 0], q[:, 1])))),
        state_generator_max_abs=float(np.max(abs(u.detach().numpy().ravel()-generator.exact_u(q[:, 0], q[:, 1])))),
    )
    qi = q.copy(); qi[:, 1] = 0.
    checks["ic_max_abs"] = float(np.max(abs(generator.exact_u(*qi.T)-problem.physics.initial_conditions()[0].target(qi))))
    for i, value in enumerate([-1., 1.]):
        qb = q.copy(); qb[:, 0] = value
        checks[f"bc_{i}_max_abs"] = float(np.max(abs(generator.exact_u(*qb.T)-problem.physics.boundary_conditions()[i].target(qb))))
    x_star, t_star = .1, .7
    u_star = float(generator.exact_u(x_star, t_star))
    f_star = float((1-(u_star-generator.background(x_star,t_star))/.03)**2)
    result = dict(paper_problem="F6", code_problem="f06", constants=expected,
                  dtype="float64", seed=seed, n_points=n_points, threshold=1e-12,
                  checks=checks, continuous_reference=dict(decision=[x_star,t_star], state=u_star, objective=f_star,
                  constraint_violation=0., proof="F=(1-exp(-z^2))^2+0.2*(t-0.7)^2 >= 0. Equality requires t=0.7 and z=0, hence x=0.1. This is the unique global minimizer of the continuous manufactured field."),
                  objective_api_checked=False, dataset_metadata_checked=False,
                  limitations=["Autodiff/IC/BC checks use finite sampled points and floating-point arithmetic; the displayed algebra establishes continuous optimality.",
                               "Stored-field interpolation has nonzero error; continuous F*=0 is not a certificate for the interpolated discretization.",
                               "This check does not read or hash the full stored field, and does not regenerate any dataset."])
    if dataset is not None:
        dataset = Path(dataset)
        with np.load(dataset) as data:
            metadata = json.loads(str(data["meta_json"]))
            for key, value in [("c", .5), ("nu", .0005), ("alpha_t", .2)]:
                np.testing.assert_allclose(metadata[key], value, rtol=0., atol=1e-15)
            for key, value in [("A", .03), ("w", .10)]:
                np.testing.assert_allclose(metadata["packet"][key], value, rtol=0., atol=1e-15)
            np.testing.assert_allclose([metadata["reference_point"]["x_star"], metadata["reference_point"]["t_star"]], [.1,.7], rtol=0., atol=1e-15)
            if metadata["packet"]["x_s(t)"] != "-0.25+0.5*t":
                raise AssertionError("Packet trajectory metadata mismatch")
            np.testing.assert_array_equal(data["x"], np.linspace(-1.,1.,24553))
            np.testing.assert_array_equal(data["t"], np.linspace(0.,1.,9601))
        problem.attach_data(str(dataset))
        class ExactProvider:
            def evaluate(self, points):
                points = np.asarray(points)
                return generator.exact_u(points[:, 0], points[:, 1])
        provider = ExactProvider()
        f_actual = problem.evaluate_fitness(np.array([x_star,t_star]), provider)
        violation = problem.violation(np.array([x_star,t_star]), provider)
        np.testing.assert_allclose([f_actual, violation], [0.,0.], rtol=0., atol=1e-12)
        # Compare the real objective API with an independently reduced expression.
        # No state query uses the stored field in this analytic check.
        reduced = (1-np.exp(-((q[:,0]+.25-.5*q[:,1])/.10)**2))**2 + .2*(q[:,1]-.7)**2
        actual = np.array([problem.evaluate_fitness(row, provider) for row in q])
        checks["objective_api_reduced_formula_max_abs"] = float(np.max(abs(actual-reduced)))
        result.update(objective_api_checked=True, dataset_metadata_checked=True,
                      dataset_metadata_grid=[24553,9601], objective_api_at_reference=f_actual)
    if any(value > result["threshold"] for value in checks.values()) or abs(f_star) > 1e-12:
        raise AssertionError(checks)
    result["status"] = "PASS"
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument("--dataset", type=Path, help="Optional F6 reference field")
    inputs.add_argument("--data-dir", type=Path, help="Directory containing f06.npz")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--points", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=20260924)
    args = parser.parse_args()
    if args.points < 1:
        parser.error("--points must be positive")
    if args.output is not None and args.output.exists():
        parser.error(f"Output already exists: {args.output}")
    dataset = args.data_dir / "f06.npz" if args.data_dir is not None else args.dataset
    result = verify(args.root, dataset, args.points, args.seed)
    text = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+"\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
