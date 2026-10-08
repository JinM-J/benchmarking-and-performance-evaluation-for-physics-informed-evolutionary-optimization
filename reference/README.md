# Reference solutions: F1–F11

Each problem has one executable entry point: `f01.py` through `f11.py`.
The problem definition, reference entry point, target file and data file share
that identifier. For example, F6 uses `problems/f06.py`, `reference/f06.py`,
`reference/targets/f06.json` and `dataset/f06.npz`.

| Problem | Entry point | Computation and verification |
| --- | --- | --- |
| F1 | `f01.py` | High-precision stationary equations and inequality checks |
| F2 | `f02.py` | High-precision KKT equations for the exact-equality target |
| F3 | `f03.py` | Algebraic target substitution; optional stored-field check |
| F4 | `f04.py` | High-precision stationary equations |
| F5 | `f05.py` | High-precision active-constraint intersection |
| F6 | `f06.py` | Manufactured solution, PDE, boundary conditions and right-side 5% response optimum |
| F7 | `f07.py` | Attaining point and independent interpolation checks on the stored Poisson field |
| F8 | `f08.py` | Equality target and implemented constraint checks; optional stored-field check |
| F9 | `f09.py` | Algebraic lower bound; optional stored-field attainment calculation |
| F10 | `f10.py` | Parameter/state root solving with the stored field and discrete quadrature |
| F11 | `f11.py` | Stored-field discrete reference optimum; separate continuous-integral check |

Run from the repository root:

```bash
python reference/f01.py --output outputs/f01_reference.json
python reference/f06.py --output outputs/f06_reference.json
python reference/f07.py --data-dir dataset --output outputs/f07_reference.json
python reference/f09.py --data-dir dataset --output outputs/f09_reference.json
python reference/f10.py --data-dir dataset --output outputs/f10_reference.json
python reference/f11.py --data-dir dataset --output outputs/f11_reference.json
```

F7 requires `f07.npz`. F9's `--data-dir` option additionally checks its stored
field; without it, only the algebraic bound is checked. F10 requires `f10.npz`.
For F11, choose `--data-dir` with `f11.npz` for the discrete optimum, or
`--continuous` for the distinct continuous-integral analysis without loading data.
F3/F8 accept optional `--data-dir`; F6 accepts it to check field metadata.
Every entry supports `--help` and `--output`.

`targets/f01.json`–`targets/f11.json` contain the algebraic reference records.
`common/` contains shared calculation and reporting functions; it is not a
second set of problem entry points.

Algebraic optimality, PDE accuracy and attainability in an interpolated field
are separate claims. Root-solver decimal precision does not establish global
optimality. F2/F8 equality targets are distinct from tolerance-band optima.
Each output states the checks performed and their limits.
