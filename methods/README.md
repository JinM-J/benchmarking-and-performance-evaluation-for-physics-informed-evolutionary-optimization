# Methods and evaluation

## Included main methods

| CLI name | Model / information | Search |
| :--- | :--- | :--- |
| `gp_de` | GP fitted to state labels | DE |
| `pigp_de` | Operator-informed GP, linear PDE operator covariance blocks | DE |
| `pinn_de` | Neural state model with data/PDE/IC/BC losses | DE |
| `rbfn_de` | Gaussian radial-basis state network | DE |
| `mlp_de` | Neural state model with data loss only | DE |
| `pino_de` | Fourier-operator backbone with data/PDE/IC/BC losses | DE |
| `ji_sade_grm` | Decision-level objective/constraint cubic RBFN models | SaDE-SA-GRM adaptation |
| `glosade` | Decision-level GRNN / cubic RBF models | GLoSADE adaptation |

The cubic RBFN bundle used inside SaDE-SA-GRM is separate from `rbfn_de`'s state surrogate.

PIGP uses covariance blocks for u and its linear PDE operator. Known forcing is
available at collocation points without consuming state-label FE. Nonlinear terms
are represented through residual-block noise. PINO uses
`surrogates/fno.py` as its Fourier-operator backbone.

The field methods support multiple optimizers. Available method identifiers are
defined in `core/registry.py`. Constraint suffixes map to
`epsilon`, `feasibility` and `decode`; decode is retained as experimental code and
is not a selected main comparison. The two black-box decision-level methods have
their own search/constraint loops and do not receive PDE losses or IC/BC training
information. Do not describe their inner populations as additional charged HF
state labels.

## Budget and reporting

The `paper` preset counts **state labels**, including for the three parametric
problems. Its per-problem budgets are specified in `protocols/paper.yaml`. At each online
update the parametric problems preserve the selected complete `(x,t,parameter)`
decisions. `demo_f06` provides a reduced configuration for the installation example.

`HighFidelityOracle.n_state_queries` counts state rows requested by the algorithm.
Reference values come from stored-array interpolation. One state query is not a
fresh full PDE solve. The auxiliary `n_parameter_queries` counter sums distinct
parameter values within calls; it is not a globally deduplicated solve count.

Integral objective evaluation uses the discrete rules in [problem conventions](../problems/README.md). For an
HF archive entry, `_PointProvider` anchors its paid state label and obtains other
quadrature states from the reference interpolator outside the label counter.
Reference metrics likewise do not consume label FE. These conventions describe
the implementation; they do not establish equal full-solver cost between methods.

The reported objective comes from `archive_real_obj/archive_real_vio`, selecting
finite entries with violation ≤`1e-4`. Final-population reference values and
surrogate fitness are diagnostic. F2 computes the sum of excesses beyond two
0.1 equality bands; F8 computes the excess beyond a 0.001 band. The final `1e-4`
threshold is additional to those bands.

The aggregator reports the feasible-run objective mean and standard
deviation (`ddof=0`), separately recording total and feasible counts. A run with
no feasible archive entry has no feasible objective. Protocol and dataset hash
remain part of each group key.

## Literature baseline attribution

`methods/glosade.py` records its source: Y. Wang, D.-Q. Yin, S. Yang and G. Sun,
“Global and Local Surrogate-Assisted Differential Evolution for Expensive
Constrained Optimization Problems With Inequality Constraints,” IEEE Transactions
on Cybernetics 49(5), 1642–1656 (2019), DOI `10.1109/TCYB.2018.2809430`.
The implementation documents its small-budget and SciPy local-solver adaptations.

`methods/ji_saea.py` and `optimizers/sade_de.py` implement the Ji et al.
SaDE-SA-GRM baseline adaptation: objective/constraint cubic RBFNs, five-strategy
SaDE, constraint-consensus infill, stagnation-driven local search and an HF-charged
gradient-repair step. It does not reproduce the original paper's large expensive
initialization. See the benchmark manuscript for the complete reference.

Use the method papers and the accompanying benchmark manuscript when citing
these algorithms.
