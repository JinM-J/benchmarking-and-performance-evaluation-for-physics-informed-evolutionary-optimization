"""Refine the F5 algebraic candidate at the intersection of its constraint bounds.

This root check does not establish PDE reachability or global optimality."""
from pathlib import Path
import sys

import mpmath as mp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from reference.common.algebraic import record, run_cli


def solve():
    mp.mp.dps = 80
    def quartic(x):
        return x**4-12*x**3+40*x**2-48*x+17
    x = mp.findroot(quartic, mp.mpf("2.32952"), tol=mp.mpf("1e-70"), maxsteps=100)
    z = 2+2*x**4-8*x**3+8*x**2
    upper2 = 36+4*x**4-32*x**3+88*x**2-96*x
    result = record("f05", -(x+z), {"z1": x, "z2": z},
           [quartic(x), z-upper2], ["2.32952"], "1e-70")
    return result


if __name__ == "__main__":
    raise SystemExit(run_cli(solve, __file__, __doc__))
