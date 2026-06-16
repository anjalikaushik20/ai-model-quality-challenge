"""Delegates to pruner/validators.py — no Part-B-specific logic."""
from importlib.util import spec_from_file_location, module_from_spec
from pathlib import Path

_path = Path(__file__).parents[2] / 'pruner' / 'validators.py'
_spec = spec_from_file_location('_shared_pruner_validators', _path)
_mod = module_from_spec(_spec)
_spec.loader.exec_module(_mod)

score_preservation = _mod.score_preservation
go_nogo_agreement = _mod.go_nogo_agreement
kl_divergence_difficulty = _mod.kl_divergence_difficulty
