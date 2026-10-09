# Experiment configuration

[`paper.yaml`](paper.yaml) defines the paper experiment configuration for
F1–F11, including budgets, surrogate parameters, evaluation grids and repeat
schedules. Run commands from the repository root; use `--dry-run` to
inspect a schedule before starting experiments.

Batch reproduction uses the `main` group: 30 runs with seeds 450–479.
The reduced installation demonstration is configured separately.

## Search and state-label budgets

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

For field-surrogate methods, the budget is
`initial + floor(generations / interval) * labels_per_update`. Initial fitting
and online fits use the accumulated labels. Generations count optimizer
iterations, not network training steps. The two decision-level literature
baselines retain their own initialization and search rules under the same total
HF budget; periodic top-K does not govern their internal loops.

Labels are interpolated from stored reference fields. Generating a reference
field is a separate operation. Diagnostic reference queries do not consume the
optimization label budget. See [evaluation conventions](../methods/README.md#budget-and-reporting).

## F6

The objective targets the right-side 5% packet response. The continuous
reference point is `(0.1 + 0.1 * sqrt(log(20)), 0.7)`, with objective zero.
[The problem definition](../docs/problem_definitions.md#f6)
states the objective, derivation and interpolation distinction.

PINN, PINO and MLP use **100 steps per fit**. F6 has no additional algebraic
constraints, so the
constraint-handling comparison does not apply.

## F8

| Model | Fixed settings |
| --- | --- |
| GP | Matérn 3/2, length-scale bounds `[0.1, 100]`, no mean centering, no additional fitting restarts; noise initialized at `1e-8`, fitted within `[1e-8, 1e5]` |
| PIGP | Operator-informed RBF, length-scale bounds `[0.01, 100]`, no mean centering, 100 collocation points; learned data and operator-block noise |
| PINN | Layers `[2, 100, 100, 100, 100, 1]`, 300 steps per fit; data weight 10, PDE weight 0.001; learning rate 0.001 to 0.0002 |
| MLP | Layers `[2, 100, 100, 100, 100, 1]`, 500 steps per fit; exponential learning rate 0.001 to 0.0002 |
| PINO | Grid 64x64, modes 12x12, 32 hidden channels, 4 layers, 300 steps per fit; data weight 100, PDE weight 0.0001, IC/BC weights 1; learning rate 0.001 to 0.00001 |
| RBFN | `sigma_k=2`, `sigma_floor=0.0001`, regularization `1e-6` |

Disabling GP/PIGP mean centering retains standard-deviation scaling.
Reference interpolation extends F8's half-open spatial grid periodically and
evaluates the prescribed initial profile exactly at `t=0`. PINN/PINO train the
periodic value and first spatial derivative terms (`max_periodic_bc_order=1`).
The PDE definition also supplies higher derivatives; orders 2 and 3 are separate
diagnostics. The global MSE grid is 100x100. GP and PIGP use different kernel
families, so this configuration is not a matched-kernel physics-only ablation.

## Repeat schedules

The `run_groups` section of `paper.yaml` defines these schedules:

| Group | Problems | Seeds | Methods / purpose |
| --- | --- | --- | --- |
| `main` | F1-F11 | 450-479 | Eight common methods |
| `optimizers` | F6, F8 | 450-479 | PSO and CMA-ES with GP, PIGP, PINN, RBFN and PINO; DE results come from `main` |
| `constraints` | F8 | 450-479 | Epsilon and feasibility rules with GP, PIGP, PINN and PINO; penalty results come from `main` |
| `residuals` | F6, F8 | 481-490 | Six field models; save final models for offline residual evaluation |
| `illustrations` | F6, F8 | 480 | Six field models; save final models and 401x401 illustration fields |

Residual and illustration runs are separate from the 30-run main statistics.
Illustration-grid errors do not replace the 100x100 global MSE. Residual
checkpoints defer the final automatic residual calculation; use the
[checkpoint evaluator](../reproduction/README.md) for offline diagnostics.
The main seeds were used during configuration selection; these results are
not independent validation after tuning.

```bash
python experiments/batch.py --protocol paper --group main --problems f06,f08 --dry-run
python experiments/batch.py --protocol paper --group optimizers --dry-run
python experiments/batch.py --protocol paper --group constraints --dry-run
python experiments/batch.py --protocol paper --group residuals --dry-run
python experiments/batch.py --protocol paper --group illustrations --dry-run
```

Remove `--dry-run` to execute. Group outputs default to `results/<group>/`;
`--output-root` selects another empty location. Explicit `--methods`, `--runs`
and `--base-seed` override the schedule. Omitting `--group` with the `paper`
protocol selects the `main` group. Single-run and batch main
outputs share `results/main/`, which is also the aggregator's default.
Other protocols use a one-run batch default unless a group is selected.
Summarize the main group with:

```bash
python experiments/aggregate.py
```

Each run saves the complete `protocol_config`, applied `surrogate_config`,
actual HF counts and protocol SHA256 in `resolved_config.yaml`.
`demo_f06.yaml` is a reduced installation example.
