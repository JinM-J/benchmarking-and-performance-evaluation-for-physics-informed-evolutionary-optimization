"""Refine the F1 algebraic stationary point and check its inequalities.

This root check does not establish PDE reachability or global optimality."""
from pathlib import Path
import sys

import mpmath as mp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from reference.common.algebraic import record, run_cli


def solve():
    mp.mp.dps = 70
    def f1(a, b):
        return -mp.sin(2*mp.pi*a)**3 * mp.sin(2*mp.pi*b) / (a**3*(a+b))
    def d1a(a, b):
        return mp.diff(lambda x: f1(x, b), a)
    def d1b(a, b):
        return mp.diff(lambda x: f1(a, x), b)
    a, b = mp.findroot((d1a, d1b), (mp.mpf("1.228"), mp.mpf("4.245")),
                       tol=mp.mpf("1e-60"), maxsteps=100)
    result = record("f01", f1(a, b), {"z1": a, "z2": b},
           [d1a(a, b), d1b(a, b)], ["1.228", "4.245"], "1e-60")
    inequalities = [a*a-b+1, 1-a+(b-4)**2]
    result["inequality_residuals"] = [mp.nstr(v, 30) for v in inequalities]
    result["algebraic_feasible"] = all(v <= 0 for v in inequalities)
    return result


if __name__ == "__main__":
    raise SystemExit(run_cli(solve, __file__, __doc__))
