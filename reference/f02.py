"""Refine the F2 equality-constrained algebraic KKT candidate.

The equations impose exact equalities. The optimization experiment uses
nonzero equality tolerances; this candidate does not certify that band problem."""
from pathlib import Path
import sys

import mpmath as mp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from reference.common.algebraic import record, run_cli


def solve():
    mp.mp.dps = 70
    def equations2(a, b, c, lam, mu):
        return (-2*a-b-c+2*lam*a+8*mu, -4*b-a+2*lam*b+14*mu,
                -2*c-a+2*lam*c+7*mu, a*a+b*b+c*c-25, 8*a+14*b+7*c-56)
    initial = ["3.512", "0.217", "3.552", "0", "0"]
    a, b, c, lam, mu = mp.findroot(equations2, tuple(map(mp.mpf, initial)),
                                  tol=mp.mpf("1e-60"), maxsteps=100)
    result = record("f02", 1000-a*a-2*b*b-c*c-a*b-a*c,
           {"z1": a, "z2": b, "z3": c, "lambda": lam, "mu": mu},
           equations2(a, b, c, lam, mu), initial, "1e-60")
    result["constraint_semantics"] = "Exact equalities; not the experiment's tolerance bands."
    return result


if __name__ == "__main__":
    raise SystemExit(run_cli(solve, __file__, __doc__))
