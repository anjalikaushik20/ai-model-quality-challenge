"""Delegates to pruner/base.py — no Part-B-specific logic."""
from importlib.util import spec_from_file_location, module_from_spec
from pathlib import Path

_path = Path(__file__).parents[2] / 'pruner' / 'base.py'
_spec = spec_from_file_location('_shared_pruner_base', _path)
_mod = module_from_spec(_spec)
_spec.loader.exec_module(_mod)

BenchmarkPruner = _mod.BenchmarkPruner
