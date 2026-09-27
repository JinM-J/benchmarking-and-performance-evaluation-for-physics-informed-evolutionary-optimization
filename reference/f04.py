"""Refine the F4 algebraic stationary point.

This root check does not establish PDE reachability or global optimality."""
from pathlib import Path
import sys

import mpmath as mp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from reference.common.algebraic import record, run_cli


def solve():
    mp.mp.dps = 70
    def equations4(a, b):
        return b-8*a+16*a**3, 8*b-mp.mpf("8.4")*b**3+2*b**5+a
    a, b = mp.findroot(equations4, (mp.mpf("0.7127"), mp.mpf("-0.08984")),
                       tol=mp.mpf("1e-60"), maxsteps=100)
    value = 4*b*b-mp.mpf("2.1")*b**4+b**6/3+a*b-4*a*a+4*a**4
    result = record("f04", value, {"z1": a, "z2": b}, equations4(a, b),
           ["0.7127", "-0.08984"], "1e-60")
    return result


if __name__ == "__main__":
    raise SystemExit(run_cli(solve, __file__, __doc__))
