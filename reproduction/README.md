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
  deviation tables and convergence plots. It verifies input checksums and uses
  the consumed HF archive. `--include-supplements` includes the optimizer and
  constraint-handling comparisons.
- `reproduce_residuals.py` reconstructs PDE residual tables and optional initial/
  boundary-condition tables at their recorded resolutions.
- `evaluate_checkpoint.py` reloads a trusted saved model, checks its predictions
  and computes PDE residuals. F7 also requires `--data-dir` containing the data
  filename recorded in that snapshot. This command does not train the model.

`metadata/` contains the run indexes, checksums, plot seeds and comparison values
for the supplied artifacts. Output problem IDs use F1–F11. Reproduction uses
the configurations, problem identities and fingerprints recorded with each
artifact; these records determine which inputs belong together.

Large fields, saved models and run outputs are supplied separately from this
source repository.
