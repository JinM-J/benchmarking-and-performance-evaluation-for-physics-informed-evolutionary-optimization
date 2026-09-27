"""Explicit public and archived problem identities; never infer from colliding names."""

PAPER_NAMESPACE = "paper_f1_f11_v1"
LEGACY_NAMESPACE = "legacy_f01_f10_f12"
LEGACY_TO_PAPER = {
    "f01": "f01", "f02": "f02", "f03": "f03", "f04": "f04", "f05": "f05",
    "f12": "f06", "f06": "f07", "f07": "f08", "f08": "f09", "f09": "f10", "f10": "f11",
}
PAPER_TO_LEGACY = {current: old for old, current in LEGACY_TO_PAPER.items()}


def paper_problem_id(name, namespace):
    """Convert a declared namespace to public F1--F11 numbering."""
    if namespace not in (PAPER_NAMESPACE, LEGACY_NAMESPACE):
        raise ValueError(f"Unknown problem namespace: {namespace!r}")
    name = str(name).lower()
    if namespace == LEGACY_NAMESPACE and name in LEGACY_TO_PAPER:
        return LEGACY_TO_PAPER[name]
    if namespace == PAPER_NAMESPACE and name in PAPER_TO_LEGACY:
        return name
    raise ValueError(f"Unknown problem {name!r} in namespace {namespace!r}")
