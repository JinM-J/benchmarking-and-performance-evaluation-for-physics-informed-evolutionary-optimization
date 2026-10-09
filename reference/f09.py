#!/usr/bin/env python3
"""Verify the algebraic F9 lower bound directly from problems/f09.py.

Usage: python reference/f09.py --output f9_lower_bound.json
Requires SymPy. With --data-dir, also verify attainability in the stored field.
No PDE data are generated.
"""
import argparse
import ast
import json
import sys
from pathlib import Path

import sympy as sp


def extract_objective(source):
    """Read F09._F as exact rational algebra without executing the source file."""
    tree = ast.parse(source)
    classes = [node for node in tree.body
               if isinstance(node, ast.ClassDef) and node.name == "F09"]
    if len(classes) != 1:
        raise ValueError("Expected exactly one F09 class")
    functions = [node for node in classes[0].body
                 if isinstance(node, ast.FunctionDef) and node.name == "_F"]
    if len(functions) != 1:
        raise ValueError("Expected exactly one F09._F function")
    function = functions[0]
    if [arg.arg for arg in function.args.args] != ["u_val", "j_val"]:
        raise ValueError("Unexpected F09._F argument signature")
    u, integral = sp.symbols("u J", real=True)
    values = {"u_val": u, "j_val": integral}

    def convert(node):
        if isinstance(node, ast.Name):
            return values[node.id]
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            # Interpret decimal source literals exactly, not as binary floats.
            return sp.Rational(ast.get_source_segment(source, node).replace("_", ""))
        if isinstance(node, ast.UnaryOp):
            operand = convert(node.operand)
            if isinstance(node.op, ast.USub):
                return -operand
            if isinstance(node.op, ast.UAdd):
                return operand
        if isinstance(node, ast.BinOp):
            left, right = convert(node.left), convert(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return left / right
            if isinstance(node.op, ast.Pow):
                return left ** right
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "float" and len(node.args) == 1
                and not node.keywords):
            return convert(node.args[0])
        raise ValueError(f"Unsupported objective expression: {ast.dump(node)}")

    for index, statement in enumerate(function.body):
        if (index == 0 and isinstance(statement, ast.Expr)
                and isinstance(statement.value, ast.Constant)
                and isinstance(statement.value.value, str)):
            continue
        if (isinstance(statement, ast.Assign) and len(statement.targets) == 1
                and isinstance(statement.targets[0], ast.Name)):
            values[statement.targets[0].id] = convert(statement.value)
        elif isinstance(statement, ast.Return) and index == len(function.body) - 1:
            return convert(statement.value), u, integral
        else:
            raise ValueError(f"Unsupported objective statement: {ast.dump(statement)}")
    raise ValueError("F09._F has no final return statement")


def verify(source_path):
    """Check the source identity, positive factors and unique algebraic equality."""
    source_bytes = source_path.read_bytes()
    objective, u, integral = extract_objective(source_bytes.decode("utf-8"))
    s, q = sp.symbols("s q", real=True)
    v = integral + sp.Rational(3, 10)
    s_state, q_state = u - v + 1, 2 * u + 3 * v
    a_quadratic = 3 * s ** 2 - 20 * s + 36
    b_quadratic = 3 * q ** 2 + 2 * q + 3
    a = 1 + s ** 2 * a_quadratic
    b = 3 + (q - 3) ** 2 * b_quadratic
    a_square = 3 * (s - sp.Rational(10, 3)) ** 2 + sp.Rational(8, 3)
    b_square = 3 * (q + sp.Rational(1, 3)) ** 2 + sp.Rational(8, 3)
    differences = {
        "source_minus_factorization": sp.expand(
            objective - a.subs(s, s_state) * b.subs(q, q_state)),
        "a_quadratic_minus_completed_square": sp.expand(a_quadratic - a_square),
        "b_quadratic_minus_completed_square": sp.expand(b_quadratic - b_square),
    }
    if any(value != 0 for value in differences.values()):
        raise AssertionError(f"Symbolic identity failure: {differences}")
    solutions = sp.solve([s_state, q_state - 3], [u, integral], dict=True)
    expected = {u: sp.Integer(0), integral: sp.Rational(7, 10)}
    if solutions != [expected] or sp.simplify(objective.subs(expected)) != 3:
        raise AssertionError(f"Unexpected equality solution: {solutions}")
    return {
        "status": "PASS",
        "paper_problem": "F9",
        "code_problem": "f09",
        "source_file": "problems/f09.py",
        "sympy_version": sp.__version__,
        "arithmetic": "Exact rational algebra from decimal source literals",
        "source_objective_expression": str(objective),
        "definitions": {"v": "J + 3/10", "s": str(s_state), "q": str(q_state)},
        "factorization": {"A": str(a), "B": str(b), "F": "A*B"},
        "positive_quadratic_forms": {"A": str(a_square), "B": str(b_square)},
        "symbolic_identity_differences": {key: str(value) for key, value in differences.items()},
        "proof": (
            "Both quadratic factors are at least 8/3 for real s,q. Hence A>=1 "
            "and B>=3, so F>=3. Equality requires s=0 and q=3, uniquely giving "
            "u=0 and J=7/10."
        ),
        "algebraic_global_lower_bound": "3",
        "unique_algebraic_equality": {"u": "0", "J": "7/10"},
        "objective_at_equality": str(objective.subs(expected)),
        "pde_reachability_checked": False,
        "limitations": [
            "This proves a lower bound for the real-valued algebraic objective. "
            "A common PDE decision attaining both target states must be verified separately.",
            "The symbolic proof does not bound floating-point evaluation or PDE interpolation error.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1],
                        help="Repository root containing problems/f09.py")
    parser.add_argument("--data-dir", type=Path, help="Also solve attainability on f09.npz")
    parser.add_argument("--output", type=Path, help="Write a new JSON file; existing files are refused")
    args = parser.parse_args()
    if args.output is not None and args.output.exists():
        parser.error(f"Output already exists: {args.output}")
    result = verify(args.root / "problems" / "f09.py")
    result["pde_reachability_checked"] = args.data_dir is not None
    if args.data_dir is not None:
        sys.path.insert(0, str(args.root))
        from reference.common import integral
        result["stored_field"] = integral.solve("F9", args.data_dir)
    payload = json.dumps(result, indent=2, ensure_ascii=True) + "\n"
    if args.output is None:
        print(payload, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as handle:
            handle.write(payload)
        print(f"PASS: F9 algebraic lower bound verified; report written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
