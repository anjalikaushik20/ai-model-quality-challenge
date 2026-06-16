"""
Load evalscope predictions and reviews JSONL files into score matrices.

File naming convention (evalscope cache format):
  predictions/<benchmark>__<model>.jsonl
  reviews/<benchmark>__<model>.jsonl

Prediction row:  {"index": int, "messages": [...], "metadata": {...}, ...}
Review row:      {"index": int, "sample_score": {"score": {"value": {"acc": float}}}}
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_SCORE_KEYS: Dict[str, Tuple[str, ...]] = {
    'live_code_bench_v5': ('acc', 'pass'),
    'aa_lcr': ('acc',),
}


def _score_from_row(row: dict, keys: Tuple[str, ...]) -> Optional[float]:
    try:
        value_dict = row['sample_score']['score']['value']
        for k in keys:
            if k in value_dict:
                v = value_dict[k]
                return float(bool(v)) if isinstance(v, bool) else float(v)
    except (KeyError, TypeError):
        pass
    for k in keys:
        if k in row:
            try:
                v = row[k]
                return float(bool(v)) if isinstance(v, bool) else float(v)
            except (TypeError, ValueError):
                pass
    return None


def _explanation_from_row(row: dict) -> str:
    try:
        return row['sample_score']['score'].get('explanation', '') or ''
    except (KeyError, TypeError):
        return ''


def load_reviews(
    reviews_dir: str,
    benchmark: str,
    models: List[str],
) -> Tuple[Dict[str, Dict[int, float]], Dict[str, Dict[int, str]]]:
    """
    Returns:
        scores:       {model: {index: float}}
        explanations: {model: {index: str}}
    """
    reviews_dir = Path(reviews_dir)
    keys = _SCORE_KEYS.get(benchmark, ('acc', 'pass'))

    scores: Dict[str, Dict[int, float]] = {}
    explanations: Dict[str, Dict[int, str]] = {}

    for model in models:
        filepath = reviews_dir / f'{benchmark}__{model}.jsonl'
        if not filepath.exists():
            raise FileNotFoundError(f'Review file not found: {filepath}')

        model_scores: Dict[int, float] = {}
        model_exps: Dict[int, str] = {}

        with open(filepath) as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as e:
                    logger.warning(f'{filepath}:{lineno} JSON error: {e}')
                    continue

                idx = int(row['index'])
                s = _score_from_row(row, keys)
                if s is not None:
                    model_scores[idx] = s
                model_exps[idx] = _explanation_from_row(row)

        scores[model] = model_scores
        explanations[model] = model_exps
        logger.info(f'Loaded {len(model_scores)} scores  {model} / {benchmark}')

    return scores, explanations


def load_predictions(
    predictions_dir: str,
    benchmark: str,
    models: List[str],
) -> Dict[str, Dict[int, dict]]:
    """
    Returns:
        {model: {index: metadata_dict}}

    The first message content is stored as '_query_text' for embedding.
    """
    predictions_dir = Path(predictions_dir)
    metadata_by_model: Dict[str, Dict[int, dict]] = {}

    for model in models:
        filepath = predictions_dir / f'{benchmark}__{model}.jsonl'
        if not filepath.exists():
            raise FileNotFoundError(f'Prediction file not found: {filepath}')

        model_meta: Dict[int, dict] = {}

        with open(filepath) as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as e:
                    logger.warning(f'{filepath}:{lineno} JSON error: {e}')
                    continue

                idx = int(row['index'])
                meta = dict(row.get('metadata') or {})

                messages = row.get('messages', [])
                if messages:
                    first = messages[0]
                    content = first.get('content', '') if isinstance(first, dict) else str(first)
                    meta['_query_text'] = content[:3000]

                model_meta[idx] = meta

        metadata_by_model[model] = model_meta

    return metadata_by_model


def build_score_matrix(
    scores: Dict[str, Dict[int, float]],
    all_indices: Optional[List[int]] = None,
) -> pd.DataFrame:
    """(n_models × n_items) DataFrame. Rows = models, columns = item indices."""
    if all_indices is None:
        all_indices = sorted({idx for ms in scores.values() for idx in ms})

    data = {
        model: [scores[model].get(idx, np.nan) for idx in all_indices]
        for model in scores
    }
    return pd.DataFrame(data, index=all_indices).T


def build_item_metadata(
    metadata_by_model: Dict[str, Dict[int, dict]],
    all_indices: List[int],
) -> pd.DataFrame:
    """Per-item metadata DataFrame from the first model's predictions."""
    first_model = next(iter(metadata_by_model))
    meta_dict = metadata_by_model[first_model]

    rows = [{'index': idx, **(meta_dict.get(idx) or {})} for idx in all_indices]
    df = pd.DataFrame(rows)
    if 'index' in df.columns:
        df = df.set_index('index')
    return df
