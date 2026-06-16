"""Delegates to pruner/irt.py — no Part-B-specific logic."""
from importlib.util import spec_from_file_location, module_from_spec
from pathlib import Path

_path = Path(__file__).parents[2] / 'pruner' / 'irt.py'
_spec = spec_from_file_location('_shared_pruner_irt', _path)
_mod = module_from_spec(_spec)
_spec.loader.exec_module(_mod)

assign_strata = _mod.assign_strata
assign_strata_from_labels = _mod.assign_strata_from_labels
compute_pass_rates = _mod.compute_pass_rates
