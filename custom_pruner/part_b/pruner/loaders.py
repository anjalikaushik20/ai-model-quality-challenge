"""
MMMU-specific data loaders for the Part B benchmark pruner.

MMMU reviews are organised per-subject under reviews/<model>/mmmu_<Subject>.jsonl.

Each review line schema:
    {
      "index": int,
      "input": "<question prompt — may contain base64 image blobs>",
      "target": "<correct answer letter>",
      "sample_score": {
        "score": {"value": {"acc": 1.0}, "extracted_prediction": "A", ...},
        "sample_metadata": {
          "id": "validation_Biology_25",   ← globally-unique item ID
          "subfield": "Ecology",
          "img_type": "['Plots and Charts']",
          "topic_difficulty": "Medium"
        }
      }
    }
"""

import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_BASE64_RE = re.compile(r'data:image/[a-zA-Z]+;base64,[A-Za-z0-9+/=\r\n]+')

# Six mutually-exclusive image-type groups used as multi-hot clustering features.
# Grouping keeps semantically similar types together so the 6-dim vector
# drives clustering toward image-type diversity rather than collapsing all
# visual items into one undifferentiated cluster.
_N_IMG_TYPE_GROUPS = 6

_IMG_TYPE_GROUPS: Dict[str, int] = {
    # 0 — Scientific / technical diagrams
    'Diagrams': 0,
    'Technical Blueprints': 0,
    'Trees and Graphs': 0,
    'DNA Sequences': 0,
    'Chemical Structures': 0,
    'Geometric Shapes': 0,
    # 1 — Charts / quantitative data
    'Plots and Charts': 1,
    'Mathematical Notations': 1,
    # 2 — Medical / microscopy
    'Medical Images': 2,
    'Pathological Images': 2,
    'Microscopic Images': 2,
    'Body Scans: MRI, CT scans, and X-rays': 2,
    # 3 — Photography / artistic
    'Photographs': 3,
    'Portraits': 3,
    'Landscapes': 3,
    'Paintings': 3,
    'Sculpture': 3,
    'Comics and Cartoons': 3,
    'Poster': 3,
    'Logos and Branding': 3,
    'Screenshots': 3,
    'Icons and Symbols': 3,
    # 4 — Geographic / spatial
    'Maps': 4,
    # 5 — Tabular / text-heavy / other
    'Tables': 5,
    'Sketches and Drafts': 5,
    'Other': 5,
}


def strip_base64(text: str) -> str:
    """Replace inline base64 image data with a short placeholder."""
    return _BASE64_RE.sub('[IMAGE]', text)


def img_type_to_multihot(img_type_str: str) -> np.ndarray:
    """
    Convert an img_type string (e.g. \"['Plots and Charts', 'Maps']\") to a
    6-dimensional multi-hot vector over _IMG_TYPE_GROUPS.

    Items that don't match any known type fall into group 5 (other/tabular).
    An empty or unrecognised string also maps to group 5.
    """
    result = np.zeros(_N_IMG_TYPE_GROUPS, dtype=float)
    items = re.findall(r"'([^']+)'", img_type_str or '')
    if not items:
        result[5] = 1.0
        return result
    for item in items:
        result[_IMG_TYPE_GROUPS.get(item, 5)] = 1.0
    return result


def load_mmmu_reviews(
    reviews_dir: str,
    model: str,
) -> Tuple[Dict[str, float], List[str], pd.DataFrame]:
    """
    Load all scored MMMU items from per-subject JSONL files.

    Args:
        reviews_dir: path to the directory containing <model>/ subdirectories
        model:       model name (e.g. 'glm-4.5v-fp8')

    Returns:
        scores:   {item_id → acc (0.0 or 1.0)}
        item_ids: sorted list of globally-unique item ID strings
        meta_df:  DataFrame indexed by item_id with columns:
                    subject, subfield, img_type, topic_difficulty, _query_text
    """
    model_dir = Path(reviews_dir) / model
    if not model_dir.exists():
        raise FileNotFoundError(f'MMMU reviews directory not found: {model_dir}')

    scores: Dict[str, float] = {}
    meta_rows = []

    for filepath in sorted(model_dir.glob('mmmu_*.jsonl')):
        subject = filepath.stem[len('mmmu_'):]
        logger.info(f'Loading subject {subject!r}')

        with open(filepath) as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    logger.warning(f'{filepath}:{lineno} JSON error: {exc}')
                    continue

                sample_score = row.get('sample_score', {})
                sm = sample_score.get('sample_metadata', {})

                item_id: str = sm.get('id') or f'{subject}_{row["index"]}'

                acc = (
                    sample_score
                    .get('score', {})
                    .get('value', {})
                    .get('acc', None)
                )
                if acc is not None:
                    scores[item_id] = float(acc)

                raw_text: str = row.get('input', '')
                clean_text = strip_base64(raw_text)[:3000]

                meta_rows.append({
                    '_item_id': item_id,
                    'subject': subject,
                    'subfield': sm.get('subfield', ''),
                    'img_type': sm.get('img_type', '[]'),
                    'topic_difficulty': sm.get('topic_difficulty', 'Medium'),
                    '_query_text': clean_text,
                })

    if not meta_rows:
        raise RuntimeError(f'No MMMU review data found under {model_dir}')

    meta_df = pd.DataFrame(meta_rows).set_index('_item_id')
    item_ids = sorted(scores.keys())
    logger.info(f'Loaded {len(item_ids)} MMMU items for model {model!r}')
    return scores, item_ids, meta_df


def build_mmmu_score_matrix(
    scores: Dict[str, float],
    model: str,
    item_ids: List[str],
) -> pd.DataFrame:
    """Return a (1 × n_items) DataFrame — one row per model, columns = item IDs."""
    data = {model: [scores.get(idx, np.nan) for idx in item_ids]}
    return pd.DataFrame(data, index=item_ids).T
