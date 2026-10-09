"""Export the actual final surrogate field without retraining or HF calls."""
import json
from pathlib import Path

import numpy as np


def export_final_field(problem, method, protocol, data_path, result,
                       out_dir, grid_n, run_manifest, preserve_checkpoint=False):
    from evaluation.metrics.grid import MetricGrid

    out_dir = Path(out_dir)
    bounds = np.asarray(problem.query_bounds, dtype=float)
    if bounds.shape not in ((2, 2), (3, 2)):
        raise ValueError("Only 2D state fields or fixed-parameter slices of 3D fields are supported")
    from experiments.surrogate_checkpoint import save_checkpoint
    checkpoint = "final_surrogate.pkl"
    if preserve_checkpoint:
        if not (out_dir / checkpoint).is_file():
            raise FileNotFoundError(out_dir / checkpoint)
    else:
        save_checkpoint(method.surrogate, out_dir / checkpoint, run_manifest["problem"],
                        metadata=dict(method=run_manifest["method"], seed=run_manifest["seed"],
                                      dataset=run_manifest["dataset"], offline_physics_profile="native"))
    reference = problem.load_reference(data_path)
    coordinate_names = list(problem.physics.coordinate_names)
    slice_info = None
    if bounds.shape[0] == 3:
        # Preserve periodic endpoints, float32 handling, and query mappings from the monitor grid.
        x, t, _ = problem.metric_axes_3d(reference, grid_n, grid_n, 2)
        slice_value = float(np.mean(bounds[2]))
        slice_info = dict(axis=coordinate_names[2], value=slice_value,
                          rule="midpoint of declared query bounds; common to all methods")
    else:
        x = np.linspace(*bounds[0], grid_n)
        t = np.linspace(*bounds[1], grid_n)
    X, T = np.meshgrid(x, t, indexing="ij")
    Q = np.column_stack([X.ravel(), T.ravel()])
    if slice_info is not None:
        Q = np.column_stack([Q, np.full(len(Q), slice_value)])
        Q = np.asarray(problem.normalize_queries(Q), dtype=np.float32)
    valid = np.asarray(problem.is_valid_query(Q), dtype=bool)
    truth = np.full(len(Q), np.nan)
    prediction = np.full(len(Q), np.nan)
    truth[valid] = reference.query(Q[valid])
    if slice_info is not None:
        truth[valid] = truth[valid].astype(np.float32)
    # Chunk predictions to limit GPU memory; reference values use official NPZ interpolation.
    indices = np.flatnonzero(valid)
    for start in range(0, len(indices), 8192):
        block = indices[start:start + 8192]
        prediction[block] = method.surrogate.predict(Q[block])
    if not np.isfinite(truth[valid]).all() or not np.isfinite(prediction[valid]).all():
        raise FloatingPointError("Nonfinite reference values or predictions in the valid plotting domain")
    error = prediction - truth
    shape = X.shape
    np.savez_compressed(out_dir / "final_surrogate_field.npz",
                        x=x, t=t, u_reference=truth.reshape(shape),
                        u_prediction=prediction.reshape(shape),
                        abs_error=np.abs(error).reshape(shape),
                        valid=valid.reshape(shape), query_points=Q)

    metric = MetricGrid(problem, reference, protocol.mse_grid_nx, protocol.mse_grid_nt,
                        protocol.mse_grid_nmu)
    protocol_mse = metric.mse(method.surrogate)
    recorded_mse = float(result.db_history[-1]["mse200"])

    metadata = dict(
        purpose="single final-model illustration; excluded from formal aggregate",
        problem=run_manifest["problem"], method=run_manifest["method"],
        seed=run_manifest["seed"], protocol=run_manifest["protocol"],
        dataset=run_manifest["dataset"], grid_shape=list(shape),
        axis_order=coordinate_names, slice=slice_info,
        reference="official NPZ interpolation; 3D query/label rounding follows MetricGrid",
        prediction="actual final surrogate after the last online fit; no refit",
        error_definition="abs(u_prediction - u_reference)",
        plot_grid_mse=float(np.mean(error[valid] ** 2)),
        plot_grid_max_abs_error=float(np.max(np.abs(error[valid]))),
        protocol_grid_mse=protocol_mse, last_snapshot_mse=recorded_mse,
        last_snapshot_generation=int(result.db_history[-1]["gen"]),
        hf_queries=int(result.n_hf_queries), export_additional_hf_queries=0,
        export_additional_train_calls=0, checkpoint=checkpoint,
        recovered_export_from_saved_model=preserve_checkpoint,
    )
    (out_dir / "field_export.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(f"[FIELD] {grid_n}x{grid_n}, MSE={metadata['plot_grid_mse']:.8g}")
