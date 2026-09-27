"""Save final inference state without optimizer history or the full reference dataset."""
import importlib
import pickle
from pathlib import Path

import numpy as np
import torch
from core.numbering import PAPER_NAMESPACE, LEGACY_NAMESPACE, paper_problem_id


def save_checkpoint(surrogate, path, problem_name, *, metadata=None):
    problem_name = paper_problem_id(problem_name, PAPER_NAMESPACE)
    name = type(surrogate).__name__
    if name in {"GPSurrogate", "RBFNSurrogate"}:
        state = dict(vars(surrogate))
    elif name == "PIGPSurrogate":
        keys = ("qmin", "qmax", "_theta", "_alpha", "_Xu", "_XL", "_m_u", "_s_u",
                "predict_float32", "_fitted")
        state = {key: getattr(surrogate, key) for key in keys}
    elif name in {"PINNSurrogate", "MLPSurrogate"}:
        keys = ("layers", "_dim", "normalize_inputs", "input_axes", "output_offset",
                "output_scale", "_qmin", "_qspan")
        state = {key: getattr(surrogate, key) for key in keys if hasattr(surrogate, key)}
    elif name == "PINOSurrogate":
        keys = ("grid_shape", "n_modes", "hidden_channels", "n_layers", "projection_channels",
                "domain_padding", "output_offset", "output_scale", "parameter_batch_size",
                "_dim", "_grid_axes", "_conditioning_axes", "_n_parameters",
                "_qmin_np", "_qspan_np", "_coordinate_grid",
                "_axis_to_query", "_axis_to_grid", "_required_derivatives",
                "fourier_axes", "residual_axis_margins", "residual_scale")
        state = {key: getattr(surrogate, key) for key in keys if hasattr(surrogate, key)}
    else:
        raise TypeError(f"Inference checkpoint format is undefined for surrogate: {name}")
    state = {key: value.detach().cpu().clone() if isinstance(value, torch.Tensor) else value
             for key, value in state.items()}
    model = getattr(surrogate, "model", None)
    weights = ({key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
               if isinstance(model, torch.nn.Module) else None)
    payload = dict(version=3, problem_namespace=PAPER_NAMESPACE,
                   module=type(surrogate).__module__, class_name=name,
                   problem=problem_name, state=state, weights=weights,
                   metadata=metadata or {},
                   purpose="final prediction and offline derivative evaluation; not training resume")
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(path)


def load_checkpoint(path, device="cpu", *, problem=None):
    """Load a trusted snapshot produced by this project, optionally changing device, without refitting."""
    with Path(path).open("rb") as stream:
        payload = pickle.load(stream)
    version = payload["version"]
    if version in (1, 2):
        namespace = payload.get("problem_namespace", LEGACY_NAMESPACE)
        if namespace != LEGACY_NAMESPACE:
            raise ValueError("Checkpoint versions 1/2 must use legacy problem numbering")
    elif version == 3:
        namespace = payload.get("problem_namespace")
        if namespace != PAPER_NAMESPACE:
            raise ValueError("Checkpoint version 3 requires the public problem namespace")
    else:
        raise ValueError(f"Unsupported checkpoint version: {version!r}")
    problem_name = paper_problem_id(payload["problem"], namespace)
    if problem is not None and problem.name.lower() != problem_name:
        raise ValueError(f"Checkpoint describes {problem_name}, but received {problem.name}")
    cls = getattr(importlib.import_module(payload["module"]), payload["class_name"])
    surrogate = cls.__new__(cls)
    surrogate.__dict__.update(payload["state"])
    if payload["weights"] is not None:
        if payload["class_name"] == "PINOSurrogate":
            from surrogates.fno import FNO2d
            surrogate.model = FNO2d(2 + surrogate._n_parameters, surrogate.hidden_channels,
                                    *surrogate.n_modes, surrogate.n_layers,
                                    surrogate.projection_channels, surrogate.domain_padding)
        else:
            from surrogates.pinn import PINNNet
            surrogate.model = PINNNet(surrogate.layers)
        surrogate.model.load_state_dict(payload["weights"], strict=True)
        surrogate.model.eval()
        surrogate.device = torch.device(device)
        surrogate.model.to(surrogate.device)
        for key, value in tuple(vars(surrogate).items()):
            if isinstance(value, torch.Tensor):
                setattr(surrogate, key, value.to(surrogate.device))
    if payload["class_name"] == "PIGPSurrogate":
        from core.registry import make_problem
        from surrogates.pigp import normalize_operator, OperatorKernel
        problem = make_problem(problem_name) if problem is None else problem
        terms, _ = normalize_operator(problem.physics.linear_operator(),
                                       surrogate.qmin, surrogate.qmax)
        surrogate.kernel = OperatorKernel(terms)
    if problem is not None:
        # Callers attach data for physics evaluation; loading does not read large datasets or rerun setup.
        surrogate.problem = problem
    return surrogate


def verify_checkpoint(path, queries, predictions, device="cpu", batch_size=8192):
    model = load_checkpoint(path, device=device)
    queries = np.asarray(queries)
    replay = np.concatenate([model.predict(queries[i:i + batch_size])
                             for i in range(0, len(queries), batch_size)])
    # Match the saved model device, ordering, and chunking to avoid GPU batch-shape rounding differences.
    np.testing.assert_allclose(replay, predictions, rtol=1e-5, atol=2e-6)
    return float(np.max(np.abs(replay - predictions)))


def audit_final_checkpoint(problem, surrogate, out_dir, manifest, result):
    """Check independent snapshot replay on shared geometric probes without fitting or HF calls."""
    import hashlib
    import importlib.metadata
    import inspect
    import json

    out_dir = Path(out_dir)
    bounds = np.asarray(problem.query_bounds, dtype=float)
    axes = [lo + (hi - lo) * (np.arange(n) + .5) / n
            for (lo, hi), n in zip(bounds, (7, 9, 3))]
    queries = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, len(bounds))
    queries = queries[np.asarray(problem.is_valid_query(queries), dtype=bool)]
    assert len(queries) > 0
    predictions = surrogate.predict(queries)
    if not np.isfinite(predictions).all():
        raise FloatingPointError("Final-model probe predictions are nonfinite; retain the model and mark verification as failed")
    device = str(getattr(surrogate, "device", "cpu"))
    checkpoint = out_dir / "final_surrogate.pkl"
    replay_error = verify_checkpoint(checkpoint, queries, predictions, device=device)
    np.savez_compressed(out_dir / "checkpoint_probes.npz", queries=queries, predictions=predictions)
    root = Path(__file__).resolve().parents[1]
    sources = [Path(__file__), root / "experiments/run.py", root / "core/experiment.py",
               Path(inspect.getfile(type(surrogate))), Path(inspect.getfile(type(problem)))]
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    record = dict(status="PASS", problem=manifest["problem"], problem_namespace=PAPER_NAMESPACE,
                  checkpoint_version=3, method=manifest["method"],
                  seed=manifest["seed"], dataset=manifest["dataset"],
                  checkpoint=checkpoint.name, checkpoint_sha256=sha(checkpoint),
                  checkpoint_bytes=checkpoint.stat().st_size,
                  probes_sha256=sha(out_dir / "checkpoint_probes.npz"),
                  resolved_config_sha256=sha(out_dir / "resolved_config.yaml"),
                  hf_budget=manifest["budget"]["hf_budget_protocol"],
                  hf_queries=int(result.n_state_queries), train_calls=int(result.surrogate_train_calls),
                  last_update_generation=int(result.db_history[-1]["gen"]),
                  prediction_replay_max_abs_diff=replay_error, replay_device=device,
                  additional_hf_queries=0, additional_train_calls=0,
                  included_in_original_main_statistics=False,
                  residual_status=manifest.get("supplement", {}).get("pde_residual_status", "not_part_of_snapshot_audit"),
                  source_sha256={str(p.resolve().relative_to(root)):sha(p) for p in sources},
                  environment={name: importlib.metadata.version(name)
                               for name in ("numpy", "scipy", "scikit-learn", "torch")})
    temporary = out_dir / "checkpoint.json.tmp"
    temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(out_dir / "checkpoint.json")
    print(f"[MODEL] Final model saved and replay verified; maximum prediction difference {replay_error:.3g}")
