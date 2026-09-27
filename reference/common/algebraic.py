"""Record and write single-problem algebraic refinement results."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys

import mpmath as mp

ROOT = Path(__file__).resolve().parents[2]


def record(code, value, coordinates, residuals, initial, tolerance):
    source = ROOT / "reference/targets" / f"{code}.json"
    target = json.loads(source.read_text())
    error = abs(value - mp.mpf(target["f_star_decimal"]))
    max_residual = max(abs(v) for v in residuals)
    # Stored decimal digits do not certify PDE discretization accuracy.
    passed = error < mp.mpf("1e-45") and max_residual < mp.mpf("1e-55")
    return {
        "paper_problem": f"F{int(code[1:])}", "code_problem": code,
        "working_decimal_digits": mp.mp.dps, "initial_guess": initial,
        "findroot_tolerance": tolerance, "maxsteps": 100,
        "objective": mp.nstr(value, 65),
        "algebraic_coordinates": {k: mp.nstr(v, 65) for k, v in coordinates.items()},
        "equation_residuals": [mp.nstr(v, 15) for v in residuals],
        "archive_objective_abs_error": mp.nstr(error, 15),
        "matches_archived_value_and_equations": bool(passed),
        "pde_reachability_checked": False, "global_optimality_certified": False,
        "reference_source": str(source.relative_to(ROOT)),
        "archive_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    }


def run_cli(solve, script, description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--output", type=Path,
                        help="Write a new JSON file; otherwise print JSON to stdout")
    args = parser.parse_args()
    if args.output and args.output.exists():
        parser.error("Output exists; choose a different file.")
    result = solve()
    passed = result["matches_archived_value_and_equations"] and result.get("algebraic_feasible", True)
    result.update(
        scope="Algebraic root refinement; no PDE solve or global optimality certificate.",
        command=sys.argv, python=platform.python_version(), mpmath=mp.__version__,
        script_sha256=hashlib.sha256(Path(script).read_bytes()).hexdigest(),
        helper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        status="PASS" if passed else "FAIL", passed=passed)
    payload = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)
        print(f"{result['status']}: {result['paper_problem']} algebraic refinement; {args.output}")
    else:
        print(payload, end="")
    return 0 if passed else 1
