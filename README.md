# Benchmarking and Performance Evaluation for Physics-Informed Evolutionary Optimization

This repository provides a benchmark test suite for **physics-informed
evolutionary optimization of expensive constrained optimization problems (ECOPs)
with PDE-based constraints**. It contains eleven benchmark test functions,
F1–F11, together with multiple surrogate models, optimizers,
constraint-handling strategies, reference-solution calculations and experiment
reproduction scripts.

Physics is primarily incorporated through surrogate modeling to support
evolutionary optimization.
The problems combine PDE states with algebraic objectives and constraints,
covering PDE dynamics, algebraic landscape characteristics, state–decision
coupling and feasible region geometry.

[Test suite](#test-suite-f1f11) · [Quick start](#quick-start) ·
[Experiment configuration](#experiment-configuration) ·
[Evaluation](#performance-evaluation-and-comparison) ·
[Reference solvers](#reference-pde-solvers-and-stored-grids) ·
[Documentation](#documentation)

![Illustration of feasible region geometry under different constraints](docs/figures/feasible_region_geometry.png)

*Manuscript illustration of a PDE state surface combined with different algebraic
and geometric constraints.*

## Benchmark test suite design

The benchmark test functions are characterized along four complementary axes,
representing sources of complexity arising from PDE-based constraints,
algebraic objective landscapes and their interactions:

- **Type of PDE Dynamics:** diffusion-dominated, reaction–diffusion,
  convection–diffusion, wave-propagation, elliptic-equilibrium, and high-order
  nonlinear dissipative systems.
- **Type of Algebraic Functions:** quadratic, fractional oscillatory, Rastrigin,
  six-hump camel, Branin, Goldstein–Price, Rosenbrock, and Ackley landscapes.
- **State-Decision Coupling Mode:** fixed PDE solution fields in F1–F8 and
  parameter-dependent solution fields with integral-type objectives in F9–F11.
- **Feasible Region Geometry:** no additional algebraic constraints, inequality
  constraints, low-dimensional manifolds induced by equality constraints, and
  disconnected or highly complex feasible domains.

The [benchmark design document](docs/benchmark_design.md) gives the unified
mathematical formulation and the manuscript's detailed explanation of these axes.

## Test suite: F1–F11

| Problem | PDE model | Decision variables | Objective / constraints |
| --- | --- | --- | --- |
| [F1](docs/problem_definitions.md#f1) | Viscous Burgers | `(x, t)` | Sinusoidal fractional objective; two inequalities |
| [F2](docs/problem_definitions.md#f2) | Wave | `(x, t, z)` | Quadratic objective; two equality tolerance bands |
| [F3](docs/problem_definitions.md#f3) | Allen–Cahn-type reaction–diffusion | `(x, t)` | Rastrigin-type objective |
| [F4](docs/problem_definitions.md#f4) | Forced heat | `(x, t)` | Six-hump camel objective |
| [F5](docs/problem_definitions.md#f5) | Forced heat | `(x, t)` | Linear objective in transformed coordinates; polynomial inequalities |
| [F6](docs/problem_definitions.md#f6) | Convection–diffusion, manufactured solution | `(x, t)` | Right-side 5% response objective |
| [F7](docs/problem_definitions.md#f7) | Steady Poisson | `(x1, x2)` | Branin-type objective; perforated spatial domain |
| [F8](docs/problem_definitions.md#f8) | Kuramoto–Sivashinsky | `(x, t)` | Quadratic objective; equality tolerance band |
| [F9](docs/problem_definitions.md#f9) | Parametric reaction–diffusion | `(x, t, mu)` | Goldstein–Price-type objective with local state and state integral |
| [F10](docs/problem_definitions.md#f10) | Parametric Burgers | `(x, t, lognu)` | Shifted Rosenbrock objective with state and dissipation |
| [F11](docs/problem_definitions.md#f11) | Parametric forced heat | `(x, t, mu)` | Ackley-type objective with state and input work |

The [full problem definitions](docs/problem_definitions.md) give all objectives,
PDEs, initial/boundary conditions, constraints and reference-solution calculations.
The [problem gallery](docs/problem_gallery.md) provides objective maps,
state-surface views and parameter slices for all eleven problems.

## Quick start

Use Python 3.11 and run commands from the repository root.

### Installation and small demo

```bash
python -m pip install -r requirements-analysis.txt
python examples/demo_f06.py --out outputs/demo_f06
```

The demo generates its own small F6 field, trains a surrogate and runs
optimization with 12 HF queries. It checks installation with a smaller grid and
budget than the paper configuration, without requiring the separately supplied
reference fields.

### Run paper experiments

Large reference fields and archived experiment outputs are supplied separately
from this source repository. Import and validate the reference fields using
[the data instructions](dataset/README.md) before running paper experiments.
The [data manifest](dataset/reference_manifest.json) records the required files,
sizes, shapes and dtypes.

```bash
python experiments/run.py --problem f01 --method gp_de --protocol paper --seed 450
python experiments/batch.py --dry-run
python experiments/aggregate.py
```

The batch command previews the default 30-run paper schedule; remove
`--dry-run` to execute it. Single runs, the main batch and aggregation share
`results/main/` by default. Use `--problems` to select a subset of problems.
Each output records the complete configuration and actual HF counts.

### Reconstruct archived results

The [reproduction scripts](reproduction/README.md) rebuild objective tables,
convergence figures and residual summaries from separately supplied archived
runs, and evaluate saved surrogate models. This workflow reads existing
artifacts rather than repeating optimization.

## Experiment configuration

### Search and state-label budgets

| Problem | Initial labels | Population | Generations | Update interval | Labels per update | Total HF labels |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| F1 | 10 | 100 | 100 | 10 | 3 | 40 |
| F2 | 15 | 100 | 150 | 10 | 3 | 60 |
| F3 | 10 | 100 | 100 | 10 | 3 | 40 |
| F4 | 10 | 50 | 100 | 10 | 3 | 40 |
| F5 | 10 | 100 | 100 | 10 | 3 | 40 |
| F6 | 20 | 100 | 100 | 10 | 5 | 70 |
| F7 | 15 | 100 | 150 | 10 | 3 | 60 |
| F8 | 100 | 100 | 150 | 10 | 3 | 145 |
| F9 | 10 | 100 | 100 | 10 | 3 | 40 |
| F10 | 10 | 100 | 100 | 10 | 3 | 40 |
| F11 | 10 | 100 | 100 | 10 | 3 | 40 |

For field-surrogate methods, the HF label budget is
`initial + floor(generations / interval) * labels_per_update`. Generations count
optimizer iterations, not network training steps. The two decision-level
literature baselines retain their own initialization and search rules under the
same total HF budget.

Surrogate settings, including kernel parameters, network architectures, training
steps, loss weights and learning rates, are specified per problem in
[`protocols/paper.yaml`](protocols/paper.yaml). The
[configuration guide](protocols/README.md) documents these settings and the
main, optimizer, constraint, residual and illustration schedules. Residual and
illustration runs are separate from the 30-run main statistics.

## Performance evaluation and comparison

The main comparison includes three data-driven surrogate models—Gaussian process
(GP), radial basis function network (RBFN) and multilayer perceptron (MLP)—and
three physics-informed surrogate models—physics-informed Gaussian process
(PIGP), physics-informed neural network (PINN) and physics-informed neural
operator (PINO). Two additional surrogate-assisted evolutionary algorithms
(SAEAs), GLoSADE and SaDE-SA-GRM, broaden the comparison. Optimizer and
constraint-handling comparisons are also provided.
Model definitions, method identifiers and literature attributions are given in
[the method documentation](methods/README.md).

A multidimensional evaluation protocol separately assesses surrogate modeling
accuracy and optimization accuracy. The reported measures address three
questions:

1. **Optimization accuracy:** what is the best feasible true objective among the
   points for which the algorithm actually obtained reference information?
2. **Constraint satisfaction and repeatability:** how often are feasible
   solutions found, and how do objective values vary across independent runs?
3. **Surrogate modeling accuracy:** how accurately does the surrogate
   approximate the field, and what PDE and initial/boundary-condition residuals
   does it produce on specified evaluation points?

Result summaries report feasible-run counts and the mean and
standard deviation of feasible objectives (`ddof=0`). Convergence trajectories
use the consumed HF archive. Surrogate predictions and unconsumed
final-population diagnostics do not determine the reported best objective.

Feasibility uses a constraint-violation threshold of `1e-4`, in addition to the
problem-specific equality bands in F2 and F8. PDE residuals and field errors are
reported at stated evaluation resolutions. These measures describe different
aspects of a method and are evaluated separately.

The configured HF budget counts **oracle state-label requests**. Reference
fields are generated separately, and state queries use their interpolators.
Diagnostic evaluations and auxiliary reference quadrature are outside this
label counter. See [the evaluation conventions](methods/README.md#budget-and-reporting)
for the exact accounting and integral-objective behavior.

## Reference solutions

Each problem has a matching reference entry point, `reference/f01.py` through
`reference/f11.py`, and a target record under `reference/targets/`. Depending on
the problem, verification uses algebraic stationary or KKT equations, analytic
bounds, a manufactured solution, or numerical root finding on the stored field
with independent interpolation and quadrature checks.

PDE accuracy, algebraic optimality and attainment of a target in the implemented
discrete problem are checked separately. Numerical root precision alone does
not certify global optimality. The individual reports describe their scope,
including the distinction between continuous and discrete integral objectives.
See [reference solutions](reference/README.md) for the per-problem commands.

Reference-value checks can be run independently of optimization:

```bash
python reference/f01.py --output outputs/f01_reference.json
```

Data requirements and verification scope vary by problem and are stated in the
[reference guide](reference/README.md).

### Reference PDE solvers and stored grids

We use the following methods to compute the reference PDE fields for F1–F11.
The stored grids give the dimensions of the archived state arrays.

| Problem | Computation | Stored grid |
| --- | --- | --- |
| F1 | Centered spatial differences, forward Euler | 6001 × 80001 |
| F2 | Centered wave scheme, CFL number 1 | 16001 × 6401 |
| F3 | Centered spatial differences, forward Euler | 20001 × 20001 |
| F4 | Centered spatial differences, forward Euler | 1201 × 640001 |
| F5 | Centered spatial differences, forward Euler | 1001 × 50001 |
| F6 | Evaluation of the prescribed solution | 24553 × 9601 |
| F7 | Nine-point Poisson stencil, conjugate gradient | 1000 × 1000 |
| F8 | Periodic spatial differences, fourth-order Runge–Kutta | 200 × 10001 |
| F9 | Split reaction steps and FFT diffusion | 500 × 2001 × 1000 |
| F10 | Rusanov flux/RK2 and sine-transform diffusion | 1000 × 2001 × 1000 |
| F11 | Split source steps and FFT diffusion | 500 × 2001 × 1000 |

For F7, both grid axes are spatial. For F9–F11, the third axis indexes the
parameter values. Stored time grids may be subsampled from the internal solver
grids. The corresponding implementations are
[`dataset/generate_f01.py`](dataset/generate_f01.py) through
[`dataset/generate_f11.py`](dataset/generate_f11.py).

See [the data guide](dataset/README.md) for array conventions, data validation
and generation requirements.

## Documentation

| Topic | Guide |
| --- | --- |
| Benchmark motivation and mathematical formulation | [Benchmark design](docs/benchmark_design.md) |
| All F1–F11 definitions and reference calculations | [Problem definitions](docs/problem_definitions.md) |
| Objective maps, state surfaces and parameter slices | [Problem gallery](docs/problem_gallery.md) |
| Model definitions, method identifiers and evaluation conventions | [Methods and evaluation](methods/README.md) |
| Budgets, model settings and repeat schedules | [Experiment configuration](protocols/README.md) |
| Reference-field import, validation and generation | [Reference data](dataset/README.md) |
| Per-problem reference-value commands and verification scope | [Reference solutions](reference/README.md) |
| Archived-result reconstruction and saved-model evaluation | [Reproduction](reproduction/README.md) |

## Repository layout

| Directory | Contents |
| --- | --- |
| `problems/` | F1–F11 PDEs, objectives and constraints |
| `dataset/` | Reference-field generation, import and validation |
| `surrogates/` | Data-driven and physics-informed surrogate models |
| `optimizers/`, `methods/` | Search algorithms and surrogate-assisted methods |
| `protocols/` | Fixed budgets, model settings, repeat schedules and demonstration configuration |
| `experiments/` | Run, batch and aggregate commands |
| `core/` | Experiment execution and component registry |
| `evaluation/` | Reference interpolation, HF-query accounting and numerical metrics |
| `reference/` | One reference-solution entry point per problem |
| `reproduction/` | Archived-result reconstruction and saved-model evaluation |
| `examples/` | Small installation demo |
| `docs/` | Manuscript-based benchmark design, full F1–F11 definitions and illustrated gallery |

Problem IDs are consistent throughout: `problems/f06.py`, `reference/f06.py`,
`reference/targets/f06.json`, `dataset/generate_f06.py` and `dataset/f06.npz`
all refer to F6. Python filenames use lowercase words and underscores.

[MIT License](LICENSE). Method-specific literature attributions are listed in
[the method documentation](methods/README.md#literature-baseline-attribution).
