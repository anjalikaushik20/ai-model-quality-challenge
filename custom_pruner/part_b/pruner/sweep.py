"""Delegates to pruner/sweep.py — no Part-B-specific logic."""
from importlib.util import spec_from_file_location, module_from_spec
from pathlib import Path

_path = Path(__file__).parents[2] / 'pruner' / 'sweep.py'
_spec = spec_from_file_location('_shared_pruner_sweep', _path)
_mod = module_from_spec(_spec)
_spec.loader.exec_module(_mod)

run_sweep   = _mod.run_sweep
find_target = _mod.find_target
plot_sweep  = _mod.plot_sweep
