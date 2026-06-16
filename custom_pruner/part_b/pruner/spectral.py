"""Delegates to pruner/spectral.py — no Part-B-specific logic."""
from importlib.util import spec_from_file_location, module_from_spec
from pathlib import Path

_path = Path(__file__).parents[2] / 'pruner' / 'spectral.py'
_spec = spec_from_file_location('_shared_pruner_spectral', _path)
_mod = module_from_spec(_spec)
_spec.loader.exec_module(_mod)

build_text_embeddings = _mod.build_text_embeddings
build_similarity_matrix = _mod.build_similarity_matrix
spectral_select = _mod.spectral_select
