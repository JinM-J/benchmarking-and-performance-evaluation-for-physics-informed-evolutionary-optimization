# core/version.py
"""Benchmark, framework, and output schema versions.

BENCHMARK_VERSION identifies the frozen paper preset and later extensions.
FRAMEWORK_VERSION tracks public interfaces, protocols, and result semantics.
SCHEMA_VERSION defines history.npz and resolved_config.yaml field semantics;
readers use it to interpret stored results."""
BENCHMARK_VERSION = "v1.1.1"
FRAMEWORK_VERSION = "1.3.2"
SCHEMA_VERSION = "1.2"
