"""
SpectralStratifiedPruner — MMMU benchmark compression (Part B).

Pipeline
--------
1. Load scored MMMU items (360 items, 1 model, 12 subjects) from per-subject
   JSONL review files.
2. Stratify by topic_difficulty metadata (Easy / Medium / Hard) — replaces
   pass_rate-based stratification since a single binary-scored model yields
   only two pass rates (0 or 1), producing no Medium stratum.
3. Allocate budget proportionally across strata (mirrors full difficulty
   distribution in the pruned set).
4. Within each stratum, run spectral clustering on:
     • text embeddings (sentence-transformer, base64 stripped)
     • subject one-hot features (drives cross-subject diversity)
     • visual_score and topic_difficulty numeric features
5. Pick one representative per cluster using centrality (closest to centroid).
   IRT discrimination is undefined with a single model, so centrality is used
   for all three strata.
6. Output selected item IDs and run the validation suite.

evalscope integration
---------------------
    from pruner import SpectralStratifiedPruner, PrunerConfig
    config  = PrunerConfig(benchmark='mmmu', models=['glm-4.5v-fp8'], ...)
    pruner  = SpectralStratifiedPruner(config)
    pruner.fit()
    dataset = dataset.filter(pruner.evalscope_filter())
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .base import BenchmarkPruner
from .irt import (
    assign_strata,
    assign_strata_from_labels,
    compute_pass_rates,
)
from .loaders import (
    _N_IMG_TYPE_GROUPS,
    build_mmmu_score_matrix,
    img_type_to_multihot,
    load_mmmu_reviews,
)
from .spectral import build_text_embeddings, spectral_select
from .validators import (
    go_nogo_agreement,
    kl_divergence_difficulty,
    score_preservation,
)

logger = logging.getLogger(__name__)

_STRATUM_EASY = 'easy'
_STRATUM_MEDIUM = 'medium'
_STRATUM_HARD = 'hard'


@dataclass
class PrunerConfig:
    """Configuration for SpectralStratifiedPruner."""

    benchmark: str
    """Benchmark name — use 'mmmu' for MMMU."""

    models: List[str]
    """Model identifiers.  For MMMU: ['glm-4.5v-fp8']."""

    evals_dir: str
    """Path to the directory containing 'predictions/' and 'reviews/'."""

    target_size: int
    """Desired number of items in the pruned subset."""

    embedding_model: str = 'all-MiniLM-L6-v2'
    """sentence-transformers model used to embed item question texts."""

    cache_dir: Optional[str] = None
    """Directory for caching embeddings to disk.  Auto-derived if None."""

    budget_fractions: Optional[Dict[str, float]] = None
    """
    Override per-stratum budget fractions.
    None (default) → derived from actual stratum sizes (proportional allocation).
    """

    go_nogo_threshold: Optional[float] = None
    """
    Score threshold for go/no-go decisions.
    None → calibrated automatically as the mean of full model scores.
    """


class SpectralStratifiedPruner(BenchmarkPruner):
    """
    Stratified spectral benchmark compression for MMMU (Part B).

    See module docstring for the full pipeline description.
    """

    def __init__(self, config: PrunerConfig) -> None:
        self.config = config
        self._selected: Optional[List[str]] = None
        self._fit_report: Optional[dict] = None
        self._score_matrix_cache: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------ #
    # BenchmarkPruner interface                                            #
    # ------------------------------------------------------------------ #

    def fit(
        self,
        score_matrix: Optional[pd.DataFrame] = None,
        item_texts: Optional[List[str]] = None,
        item_metadata: Optional[pd.DataFrame] = None,
    ) -> None:
        """Fit the pruner.  Loads data from evals_dir if score_matrix is None."""
        if score_matrix is None:
            score_matrix, item_texts, item_metadata = self._load_data()

        score_matrix = score_matrix.dropna(axis=1, how='all')
        all_indices: List[str] = list(score_matrix.columns)
        n_items = len(all_indices)
        n_models = len(score_matrix)

        logger.info(
            f'[{self.config.benchmark}] Fitting on {n_items} items, {n_models} model(s)'
        )

        S = score_matrix.values.astype(float)  # (n_models, n_items)
        pass_rates = compute_pass_rates(S)

        # Stratification -------------------------------------------------------
        # Prefer explicit topic_difficulty labels (MMMU) over pass_rate-derived
        # strata.  With a single binary-scored model, pass_rates are 0 or 1 and
        # produce no Medium stratum, so the metadata labels are essential.
        if (
            item_metadata is not None
            and 'topic_difficulty' in item_metadata.columns
        ):
            diff_labels = np.array([
                str(item_metadata.at[idx, 'topic_difficulty'])
                if idx in item_metadata.index else 'Medium'
                for idx in all_indices
            ])
            stratum_labels = assign_strata_from_labels(diff_labels)
        else:
            stratum_labels = assign_strata(pass_rates)

        strata: Dict[str, np.ndarray] = {
            name: np.where(stratum_labels == name)[0]
            for name in (_STRATUM_EASY, _STRATUM_MEDIUM, _STRATUM_HARD)
        }
        for name, pos in strata.items():
            pr_range = (
                f'[{pass_rates[pos].min():.2f}, {pass_rates[pos].max():.2f}]'
                if len(pos) > 0 else 'N/A'
            )
            logger.info(f'  {name:6s}: {len(pos):4d} items  pass_rate {pr_range}')

        irt_info = np.zeros(n_items)

        # Data-derived feature scales (information-theoretic) -------------------
        # Scale = log2(n_classes): weights each feature block by its maximum-
        # entropy information content (bits).  A feature with more distinct
        # classes encodes more potential diversity and receives a higher scale.
        n_unique_subjects = (
            len(item_metadata['subject'].dropna().unique())
            if item_metadata is not None and 'subject' in item_metadata.columns
            else 1
        )
        subj_scale = float(np.log2(max(2, n_unique_subjects)))
        img_scale = float(np.log2(max(2, _N_IMG_TYPE_GROUPS)))
        logger.info(
            f'  Feature scales: subject={subj_scale:.3f} (log2({n_unique_subjects})), '
            f'img_type={img_scale:.3f} (log2({_N_IMG_TYPE_GROUPS}))'
        )

        # Budget allocation ----------------------------------------------------
        budget = self._allocate_budget(strata)
        logger.info(f'  Budget: {budget}')

        # Spectral selection per stratum ----------------------------------------
        # Within each stratum the item pool is split into two sub-pools:
        #   visual  — img_type groups 0-4 (Diagrams, Charts, Medical, Photo, Maps)
        #   tabular — img_type group 5    (Tables / Other)
        # Budget is allocated proportionally to sub-pool sizes, then spectral
        selected_positions: List[int] = []

        for stratum_name, positions in strata.items():
            k = budget[stratum_name]
            if k == 0 or len(positions) == 0:
                continue

            X = self._build_feature_matrix(
                positions, item_texts, item_metadata, all_indices,
                subj_scale=subj_scale, img_scale=img_scale,
            )
            local_indices = spectral_select(
                X=X, k=k, irt_scores=irt_info[positions], criterion='centrality'
            )
            selected_positions.extend(int(positions[li]) for li in local_indices)

        self._selected = sorted(all_indices[p] for p in selected_positions)
        self._score_matrix_cache = score_matrix

        model_abilities = {
            model: round(float(np.nanmean(S[i])), 4)
            for i, model in enumerate(score_matrix.index)
        }
        self._fit_report = {
            'benchmark': self.config.benchmark,
            'full_size': n_items,
            'pruned_size': len(self._selected),
            'compression_ratio': round(len(self._selected) / n_items, 4),
            'target_size': self.config.target_size,
            'stratum_counts': {k: int(len(v)) for k, v in strata.items()},
            'stratum_budgets': budget,
            'model_abilities': model_abilities,
        }
        logger.info(
            f'Selected {len(self._selected)}/{n_items} items '
            f'({self._fit_report["compression_ratio"]:.1%} of full benchmark)'
        )

    def select(self, target_size: Optional[int] = None) -> List[str]:
        if self._selected is None:
            raise RuntimeError('Call fit() before select().')
        if target_size is not None and target_size != len(self._selected):
            logger.warning(
                f'Requested target_size={target_size} but pruner was fitted '
                f'for {len(self._selected)} items.'
            )
        return list(self._selected)

    def validate(self) -> dict:
        """Run validation suite.  LOMO and ranking are skipped for single-model."""
        if self._selected is None:
            raise RuntimeError('Call fit() before validate().')

        sm = self._score_matrix_cache
        if sm is None:
            sm, _, _ = self._load_data()

        all_indices = list(sm.columns)
        model_names = list(sm.index)
        S = sm.values.astype(float)

        selected_positions = [
            all_indices.index(idx) for idx in self._selected
            if idx in all_indices
        ]

        full_scores = {
            m: float(np.nanmean(S[i])) for i, m in enumerate(model_names)
        }
        pruned_scores = {
            m: float(np.nanmean(S[i, selected_positions]))
            for i, m in enumerate(model_names)
        }

        threshold = self.config.go_nogo_threshold
        if threshold is None:
            threshold = float(np.mean(list(full_scores.values())))

        return {
            **self._fit_report,
            'validation': {
                'score_preservation': score_preservation(full_scores, pruned_scores),
                'go_nogo': go_nogo_agreement(full_scores, pruned_scores, threshold),
                'kl_difficulty': kl_divergence_difficulty(
                    pass_rates_full=np.nanmean(S, axis=0),
                    pass_rates_pruned=np.nanmean(S[:, selected_positions], axis=0),
                ),
                'threshold': threshold,
                'note': 'ranking and LOMO omitted — undefined for a single model',
            },
        }

    def save_artifacts(self, output_dir: str) -> None:
        """Write selected_indices.json and pruning_report.json."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        selected = self.select()
        with open(out / 'selected_indices.json', 'w') as f:
            json.dump(
                {
                    'benchmark': self.config.benchmark,
                    'method': 'spectral_stratified',
                    'target_size': len(selected),
                    'selected_indices': sorted(selected),
                },
                f,
                indent=2,
            )

        if self._fit_report:
            with open(out / 'pruning_report.json', 'w') as f:
                json.dump(self._fit_report, f, indent=2)

        logger.info(f'Artifacts written to {out}')

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

    def _load_data(self) -> Tuple[pd.DataFrame, List[str], pd.DataFrame]:
        evals_dir = Path(self.config.evals_dir)
        reviews_dir = evals_dir / 'reviews'
        model = self.config.models[0]

        scores, item_ids, meta_df = load_mmmu_reviews(str(reviews_dir), model)
        score_matrix = build_mmmu_score_matrix(scores, model, item_ids)

        item_texts = [
            str(meta_df.at[idx, '_query_text'])
            if idx in meta_df.index else ''
            for idx in item_ids
        ]
        return score_matrix, item_texts, meta_df

    def _allocate_budget(self, strata: Dict[str, np.ndarray]) -> Dict[str, int]:
        total = self.config.target_size
        floor = 1
        stratum_names = (_STRATUM_EASY, _STRATUM_MEDIUM, _STRATUM_HARD)

        if self.config.budget_fractions is not None:
            fracs = self.config.budget_fractions
        else:
            n_total = sum(len(strata.get(n, [])) for n in stratum_names)
            fracs = {
                n: len(strata.get(n, [])) / n_total if n_total > 0 else 0.0
                for n in stratum_names
            }

        budgets: Dict[str, int] = {}
        for name in stratum_names:
            available = len(strata.get(name, []))
            if available == 0:
                budgets[name] = 0
                continue
            raw = round(total * fracs.get(name, 0))
            budgets[name] = max(floor, min(raw, available))

        while sum(budgets.values()) > total:
            candidates = [
                (name, b) for name, b in budgets.items()
                if b > floor and len(strata.get(name, [])) > 0
            ]
            if not candidates:
                break
            trim = max(candidates, key=lambda x: x[1])[0]
            budgets[trim] -= 1

        while sum(budgets.values()) < total:
            available = len(strata.get(_STRATUM_MEDIUM, []))
            if budgets[_STRATUM_MEDIUM] < available:
                budgets[_STRATUM_MEDIUM] += 1
            else:
                break

        return budgets

    def _build_feature_matrix(
        self,
        positions: np.ndarray,
        item_texts: Optional[List[str]],
        item_metadata: Optional[pd.DataFrame],
        all_indices: List[str],
        subj_scale: float = 1.0,
        img_scale: float = 1.0,
    ) -> np.ndarray:
        """
        Combine text embeddings, subject one-hot, and numeric metadata.

        Subject one-hot features (scaled by _SUBJECT_FEATURE_SCALE) bias the
        spectral clustering to group items by subject, so the final selection
        naturally spans diverse subjects.
        """
        parts: List[np.ndarray] = []

        # Text embeddings
        if item_texts:
            stratum_texts = [item_texts[p] for p in positions]
            if any(t.strip() for t in stratum_texts):
                cache_dir = self.config.cache_dir or str(
                    Path(self.config.evals_dir).parent / '.cache' / 'embeddings'
                )
                try:
                    emb = build_text_embeddings(
                        stratum_texts,
                        model_name=self.config.embedding_model,
                        cache_dir=cache_dir,
                    )
                    parts.append(emb)
                except Exception as exc:
                    logger.warning(f'Embedding failed: {exc}; falling back to metadata.')

        if item_metadata is not None and not item_metadata.empty:
            stratum_indices = [all_indices[p] for p in positions]

            # Subject one-hot (primary diversity axis)
            subj_feats = self._subject_onehot_features(item_metadata, stratum_indices)
            if subj_feats is not None:
                parts.append(subj_feats * subj_scale)

            # img_type multi-hot (6-dim) — secondary diversity axis
            img_feats = self._img_type_multihot_features(item_metadata, stratum_indices)
            if img_feats is not None:
                parts.append(img_feats * img_scale)

        if not parts:
            return np.arange(len(positions), dtype=float).reshape(-1, 1)

        return np.hstack(parts)

    def _subject_onehot_features(
        self,
        meta_df: pd.DataFrame,
        indices: List[str],
    ) -> Optional[np.ndarray]:
        if 'subject' not in meta_df.columns:
            return None
        all_subjects = sorted(meta_df['subject'].dropna().unique().tolist())
        if not all_subjects:
            return None

        subj_to_col = {s: i for i, s in enumerate(all_subjects)}
        result = np.zeros((len(indices), len(all_subjects)), dtype=float)
        for row_i, idx in enumerate(indices):
            if idx in meta_df.index:
                subj = meta_df.at[idx, 'subject']
                if subj in subj_to_col:
                    result[row_i, subj_to_col[subj]] = 1.0
        return result

    def _img_type_multihot_features(
        self,
        meta_df: pd.DataFrame,
        indices: List[str],
    ) -> Optional[np.ndarray]:
        """6-dim multi-hot over image-type groups for each item."""
        if 'img_type' not in meta_df.columns:
            return None
        result = np.zeros((len(indices), _N_IMG_TYPE_GROUPS), dtype=float)
        for row_i, idx in enumerate(indices):
            if idx in meta_df.index:
                result[row_i] = img_type_to_multihot(meta_df.at[idx, 'img_type'])
        return result


# Keep the Part-A class name as an alias so validators.py lazy imports still work.
SpectralIRTStratifiedPruner = SpectralStratifiedPruner
