# core/registry.py
"""Map public problem and method names to implementations and extension points."""
from pathlib import Path

import yaml

from problems.f01 import F01
from problems.f02 import F02
from problems.f03 import F03
from problems.f04 import F04
from problems.f05 import F05
from problems.f06 import F06
from problems.f07 import F07
from problems.f08 import F08
from problems.f09 import F09
from problems.f10 import F10
from problems.f11 import F11
from methods.gp_de import GPDEMethod
from methods.pigp_de import PIGPDEMethod
from methods.pinn_de import PINNDEMethod
from methods.rbfn_de import RBFNDEMethod
from methods.mlp_de import MLPDEMethod
from methods.pino_de import PINODEMethod
from methods.ji_saea import JiSADEGRMMethod
from methods.glosade import GLoSADEMethod
# Optimizer ablations retain the surrogate, sampling policy, and penalty settings.
from methods.gp_pso import GPPSOMethod
from methods.pigp_pso import PIGPPSOMethod
from methods.pinn_pso import PINNPSOMethod
from methods.rbfn_pso import RBFNPSOMethod
from methods.mlp_pso import MLPPSOMethod
from methods.pino_pso import PINOPSOMethod
from methods.gp_cmaes import GPCMAESMethod
from methods.pigp_cmaes import PIGPCMAESMethod
from methods.pinn_cmaes import PINNCMAESMethod
from methods.rbfn_cmaes import RBFNCMAESMethod
from methods.mlp_cmaes import MLPCMAESMethod
from methods.pino_cmaes import PINOCMAESMethod
from optimizers.de import DE
from optimizers.cmaes import CMAESOptimizer
from optimizers.pso import PSOOptimizer
from protocols.base import ProtocolConfig

PROBLEMS = {
    'f01': F01,
    'f02': F02,
    'f03': F03,
    'f04': F04,
    'f05': F05,
    'f06': F06,
    'f07': F07,
    'f08': F08,
    'f09': F09,
    'f10': F10,
    'f11': F11,
}

# Problem names -> reference files under dataset/.
DATA_FILES = {
    "f01": "f01.npz",
    "f02": "f02.npz",
    "f03": "f03.npz",
    "f04": "f04.npz",
    "f05": "f05.npz",
    "f06": "f06.npz",
    "f07": "f07.npz",
    "f08": "f08.npz",
    "f09": "f09.npz",
    "f10": "f10.npz",
    "f11": "f11.npz",
}

# Six field-surrogate families and two decision-level literature baselines.
METHODS = {
    'gp_de': GPDEMethod,
    'pigp_de': PIGPDEMethod,
    'pinn_de': PINNDEMethod,
    'rbfn_de': RBFNDEMethod,
    'mlp_de': MLPDEMethod,
    'pino_de': PINODEMethod,
    'ji_sade_grm': JiSADEGRMMethod,
    'glosade': GLoSADEMethod,
    'gp_pso': GPPSOMethod,
    'pigp_pso': PIGPPSOMethod,
    'pinn_pso': PINNPSOMethod,
    'rbfn_pso': RBFNPSOMethod,
    'mlp_pso': MLPPSOMethod,
    'pino_pso': PINOPSOMethod,
    'gp_cmaes': GPCMAESMethod,
    'pigp_cmaes': PIGPCMAESMethod,
    'pinn_cmaes': PINNCMAESMethod,
    'rbfn_cmaes': RBFNCMAESMethod,
    'mlp_cmaes': MLPCMAESMethod,
    'pino_cmaes': PINOCMAESMethod,
    'gp_de_eps': GPDEMethod,
    'gp_de_fs': GPDEMethod,
    'gp_de_dc': GPDEMethod,
    'pigp_de_eps': PIGPDEMethod,
    'pigp_de_fs': PIGPDEMethod,
    'pigp_de_dc': PIGPDEMethod,
    'pinn_de_eps': PINNDEMethod,
    'pinn_de_fs': PINNDEMethod,
    'pinn_de_dc': PINNDEMethod,
    'rbfn_de_eps': RBFNDEMethod,
    'rbfn_de_fs': RBFNDEMethod,
    'rbfn_de_dc': RBFNDEMethod,
    'mlp_de_eps': MLPDEMethod,
    'mlp_de_fs': MLPDEMethod,
    'mlp_de_dc': MLPDEMethod,
    'pino_de_eps': PINODEMethod,
    'pino_de_fs': PINODEMethod,
    'pino_de_dc': PINODEMethod,
}

# Constraint suffixes -> constraint_mode, resolved by make_method.
CONSTRAINT_MODE_SUFFIX = {
    "_eps": "epsilon",
    "_fs": "feasibility",
    "_dc": "decode",
}


# Optimizer registry for search-operator ablations.
# Constructors share bounds/pop_size/max_gen/seed/penalty_* and
# fitness_function/constraint/ever_gen/online_hook arguments, allowing methods
# to forward the same kwargs to make_optimizer (e.g. GP-DE -> GP-CMAES).
# Existing *_de methods select DE explicitly.
OPTIMIZERS = {
    "de": DE,
    "cmaes": CMAESOptimizer,
    "pso": PSOOptimizer,
}


def make_optimizer(name: str, **kwargs):
    if name not in OPTIMIZERS:
        raise KeyError(f"Unknown optimizer: {name}; available: {list(OPTIMIZERS)}")
    return OPTIMIZERS[name](**kwargs)


def make_problem(name: str):
    """Construct a built-in problem or load module.path:ClassName.

    The external module must be importable through sys.path and provide the
    benchmark problem interface."""
    if ":" in name:
        import importlib
        mod_name, cls_name = name.split(":", 1)
        cls = getattr(importlib.import_module(mod_name), cls_name)
        return cls()
    if name not in PROBLEMS:
        raise KeyError(f"Unknown problem: {name}; available: {list(PROBLEMS)}; "
                       f"use 'module:Class' for an external problem")
    return PROBLEMS[name]()


def make_method(name: str, protocol, seed: int):
    if name not in METHODS:
        raise KeyError(f"Unknown method: {name}; available: {list(METHODS)}")
    # Strip _eps/_fs/_dc to recover the base method and constraint mode.
    constraint_mode = "penalty"
    base_name = name
    for suffix, mode in CONSTRAINT_MODE_SUFFIX.items():
        if name.endswith(suffix):
            constraint_mode = mode
            base_name = name[:-len(suffix)]
            break
    # Included surrogate families require explicit protocol settings.
    # Strip the _de/_pso/_cmaes optimizer suffix to obtain the surrogate name.
    surr_key = next((base_name[:-len(s)] for s in ("_de", "_pso", "_cmaes")
                     if base_name.endswith(s)), base_name)
    if surr_key in ("rbfn", "mlp", "pino", "pigp") \
            and surr_key not in protocol.surrogates:
        raise KeyError(
            f"Protocol {protocol.name} has no {surr_key} hyperparameters: "
            f"the surrogate is undefined for this protocol/problem"
        )
    if base_name == "glosade" and "glosade_lambda" not in protocol.method_params:
        raise KeyError(
            f"Protocol {protocol.name} has no GLoSADE parameters: "
            "this baseline requires explicitly configured parameters"
        )
    method = METHODS[name](protocol, seed)
    # Self-managed methods define their own feasibility rule; preserve that
    # declaration instead of mislabeling resolved_config as a penalty method.
    if not getattr(method, "needs_decision_evaluator", False):
        method.constraint_mode = constraint_mode
    return method


def load_protocol(yaml_path, problem_key: str) -> ProtocolConfig:
    """Load the requested problem configuration from a protocol YAML file."""
    with open(yaml_path, "r", encoding="utf-8") as fp:
        doc = yaml.safe_load(fp)
    problems = doc.get("problems", {})
    if problem_key not in problems:
        raise KeyError(
            f"Protocol file {yaml_path} has no parameters for {problem_key}; "
            f"verify the published settings before adding them"
        )
    problem_config = dict(problems[problem_key])
    method_defaults = doc.get("method_defaults", {}) or {}
    if not isinstance(method_defaults, dict):
        raise TypeError(f"method_defaults in protocol file {yaml_path} must be a mapping")
    problem_method_params = problem_config.get("method_params", {}) or {}
    if not isinstance(problem_method_params, dict):
        raise TypeError(
            f"Protocol file {yaml_path}: {problem_key}.method_params must be a mapping"
        )
    problem_config["method_params"] = {
        **method_defaults,
        **problem_method_params,
    }
    return ProtocolConfig.from_yaml_dict(
        name=f"{doc.get('protocol', {}).get('name', 'unnamed')}/{problem_key}",
        d=problem_config,
    )
