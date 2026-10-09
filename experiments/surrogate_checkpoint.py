"""Save final inference state without optimizer history or the full reference dataset."""
import importlib
import pickle
from pathlib import Path

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


def write_checkpoint_metadata(surrogate, out_dir, manifest, result):
    """Record the saved model's inputs without replaying predictions or hashing files."""
    import importlib.metadata
    import json

    out_dir = Path(out_dir)
    checkpoint = out_dir / "final_surrogate.pkl"
    record = dict(status="saved", problem=manifest["problem"], problem_namespace=PAPER_NAMESPACE,
                  checkpoint_version=3, method=manifest["method"], seed=manifest["seed"],
                  dataset=manifest["dataset"], checkpoint=checkpoint.name,
                  offline_physics_profile="native", checkpoint_bytes=checkpoint.stat().st_size,
                  hf_budget=manifest["budget"]["hf_budget_protocol"],
                  hf_queries=int(result.n_state_queries), train_calls=int(result.surrogate_train_calls),
                  last_update_generation=int(result.db_history[-1]["gen"]),
                  replay_device=str(getattr(surrogate, "device", "cpu")),
                  additional_hf_queries=0, additional_train_calls=0,
                  environment={name: importlib.metadata.version(name)
                               for name in ("numpy", "scipy", "scikit-learn", "torch")})
    temporary = out_dir / "checkpoint.json.tmp"
    temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(out_dir / "checkpoint.json")
    print("[MODEL] Final model and metadata saved")
