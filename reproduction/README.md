# Reproduce experimental results

These commands read the separately supplied experiment artifacts. They rebuild
results without repeating optimization or generating PDE fields.

```bash
python reproduction/reproduce_optimization.py --artifacts /path/to/artifacts \
  --out outputs/optimization --include-supplements --plot
python reproduction/reproduce_residuals.py --artifacts /path/to/artifacts \
  --out outputs/residuals --include-conditions
python reproduction/evaluate_checkpoint.py --snapshot /path/to/snapshot \
  --output outputs/checkpoint_residual.json
```

- `reproduce_optimization.py` reconstructs per-run objectives, mean/standard
  deviation tables and convergence plots using the consumed HF archive.
  `--include-supplements` includes the optimizer and constraint-handling comparisons.
- `reproduce_residuals.py` reconstructs PDE residual tables and optional initial/
  boundary-condition tables at their recorded resolutions.
- `evaluate_checkpoint.py` reloads a saved model and computes PDE residuals on
  CPU by default. F7 also requires `--data-dir` containing the data filename
  recorded in that snapshot. This command does not train the model.

`metadata/` contains run indexes and plot selections for the supplied artifacts.
Run summaries use sample standard deviations (`ddof=1`), with a separate valid
count for each metric; objective statistics include only feasible runs. Sample
standard deviation is undefined for fewer than two valid runs.
`metadata/residual_expected.csv` retains the original companion-artifact cohort.
Its F6/F8 means predate the later selected batches; it is an archived reproduction
reference, and must not replace the current paper's F6/F8 results.
Output problem IDs use F1–F11. Large fields, saved models and run outputs are
supplied separately from this source repository.
