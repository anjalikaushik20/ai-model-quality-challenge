"""
SpectralIRTStratifiedPruner — main benchmark compression class.

Pipeline
--------
1. Load model × item score matrix from evalscope JSONL predictions/reviews.
2. Compute 2PL-inspired IRT scores (difficulty, discrimination, Fisher info).
3. Assign items to strata by pass-rate (easy / medium / hard).
   - Easy and hard strata keep all-pass and all-fail items;
     they are selected for content diversity, not discrimination.
   - Medium stratum is selected for IRT informativeness.
4. Allocate budget across strata (proportional with a minimum floor).
5. Within each stratum, run spectral clustering on text embeddings + metadata.
6. Pick one representative per cluster using the stratum-appropriate criterion.
7. Output selected indices and optionally run the validation suite.

evalscope integration
---------------------
    from pruner import SpectralIRTStratifiedPruner, PrunerConfig
    config  = PrunerConfig(benchmark='live_code_bench_v5', ...)
    pruner  = SpectralIRTStratifiedPruner(config)
    pruner.fit()
    dataset = dataset.filter(pruner.evalscope_filter())
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .base import BenchmarkPruner
from .irt import (
    assign_strata,
    compute_irt_info,
    compute_pass_rates,
    estimate_abilities,
    estimate_discrimination,
)
from .loaders import (
    build_item_metadata,
    build_score_matrix,
    load_predictions,
    load_reviews,
)
from .spectral import build_text_embeddings, spectral_select
from .validators import (
    go_nogo_agreement,
    kl_divergence_difficulty,
    lomo_validation,
    ranking_preservation,
    score_preservation,
)

logger = logging.getLogger(__name__)

_STRATUM_EASY = 'easy'
_STRATUM_MEDIUM = 'medium'
_STRATUM_HARD = 'hard'


@dataclass
class PrunerConfig:
    """Configuration for SpectralIRTStratifiedPruner."""

    benchmark: str
    """Benchmark name: 'live_code_bench_v5' or 'aa_lcr'."""

    models: List[str]
    """List of model identifiers that produced the eval outputs."""

    evals_dir: str
    """
    Path to the directory that contains 'predictions/' and 'reviews/'
    subdirectories — e.g. 'Evals/Part 1'.
    """

    target_size: int = 0
    """Desired items in the pruned subset.  0 = derive from target_fraction."""

    target_fraction: float = 0.20
    """Fraction of full benchmark to retain when target_size=0 (default: 20%)."""

    embedding_model: str = 'all-MiniLM-L6-v2'
    """sentence-transformers model used to embed item texts."""

    cache_dir: Optional[str] = None
    """Directory for caching embeddings to disk.  Auto-derived if None."""

    budget_fractions: Optional[Dict[str, float]] = None
    """
    Fraction of target_size to allocate to each stratum.
    If None (default), fractions are derived from actual stratum proportions
    so the pruned set mirrors the full benchmark's difficulty distribution.
    """

    go_nogo_threshold: Optional[float] = None
    """
    Score threshold for go/no-go decisions.
    If None, calibrated automatically as the mean of full model scores.
    """



class SpectralIRTStratifiedPruner(BenchmarkPruner):
    """
    Stratified IRT-Spectral benchmark compression.

    See module docstring for the full pipeline description.
    """

    def __init__(self, config: PrunerConfig) -> None:
        self.config = config
        self._selected: Optional[List[int]] = None
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

        # Drop fully-missing columns
        score_matrix = score_matrix.dropna(axis=1, how='all')
        all_indices = list(score_matrix.columns)
        n_items = len(all_indices)
        n_models = len(score_matrix)

        effective_target = (
            self.config.target_size if self.config.target_size > 0
            else max(3, round(n_items * self.config.target_fraction))
        )
        if self.config.target_size == 0:
            logger.info(
                f'  target_size auto: {self.config.target_fraction:.0%} × '
                f'{n_items} = {effective_target}'
            )

        logger.info(f'[{self.config.benchmark}] Fitting on {n_items} items, '
                    f'{n_models} models')

        S = score_matrix.values.astype(float)  # (n_models, n_items)

        # --- IRT scoring ---
        abilities = estimate_abilities(S)
        pass_rates = compute_pass_rates(S)
        discriminations = estimate_discrimination(S, abilities)
        irt_info = compute_irt_info(discriminations, pass_rates)

        # --- Stratify ---
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

        # --- Budget allocation ---
        budget = self._allocate_budget(strata, effective_target)
        logger.info(f'  Budget: {budget}')

        # --- Spectral selection per stratum ---
        selected_positions: List[int] = []

        for stratum_name, positions in strata.items():
            k = budget[stratum_name]
            if k == 0 or len(positions) == 0:
                continue

            criterion = 'irt' if stratum_name == _STRATUM_MEDIUM else 'centrality'

            X = self._build_feature_matrix(
                positions, item_texts, item_metadata, all_indices
            )

            local_indices = spectral_select(
                X=X,
                k=k,
                irt_scores=irt_info[positions],
                criterion=criterion,
            )
            selected_positions.extend(int(positions[li]) for li in local_indices)

        self._selected = sorted(all_indices[p] for p in selected_positions)
        self._score_matrix_cache = score_matrix

        self._fit_report = {
            'benchmark': self.config.benchmark,
            'full_size': n_items,
            'pruned_size': len(self._selected),
            'compression_ratio': round(len(self._selected) / n_items, 4),
            'target_size': effective_target,
            'stratum_counts': {k: int(len(v)) for k, v in strata.items()},
            'stratum_budgets': budget,
            'model_abilities': {
                model: round(float(abilities[i]), 4)
                for i, model in enumerate(score_matrix.index)
            },
        }

        logger.info(
            f'Selected {len(self._selected)}/{n_items} items '
            f'({self._fit_report["compression_ratio"]:.1%} of full benchmark)'
        )

    def select(self, target_size: Optional[int] = None) -> List[int]:
        if self._selected is None:
            raise RuntimeError('Call fit() before select().')
        if target_size is not None and target_size != len(self._selected):
            logger.warning(
                f'Requested target_size={target_size} but pruner was fitted '
                f'for {len(self._selected)} items.  Call fit() with the new '
                f'target_size for accurate results.'
            )
        return list(self._selected)

    def validate(self) -> dict:
        """Run full validation suite and return a JSON-serialisable report."""
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
            m: float(np.nanmean(S[i]))
            for i, m in enumerate(model_names)
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
                'ranking': ranking_preservation(full_scores, pruned_scores),
                'go_nogo': go_nogo_agreement(full_scores, pruned_scores, threshold),
                'kl_difficulty': kl_divergence_difficulty(
                    pass_rates_full=np.nanmean(S, axis=0),
                    pass_rates_pruned=np.nanmean(S[:, selected_positions], axis=0),
                ),
                'lomo': lomo_validation(
                    score_matrix=S,
                    model_names=model_names,
                    all_indices=all_indices,
                    config=self.config,
                ),
                'threshold': threshold,
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
                    'method': 'spectral_irt_stratified',
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
        """Load score matrix, item texts, and item metadata from evals_dir."""
        evals_dir = Path(self.config.evals_dir)
        predictions_dir = evals_dir / 'predictions'
        reviews_dir = evals_dir / 'reviews'

        scores, _ = load_reviews(
            str(reviews_dir), self.config.benchmark, self.config.models
        )
        metadata_by_model = load_predictions(
            str(predictions_dir), self.config.benchmark, self.config.models
        )

        score_matrix = build_score_matrix(scores)
        all_indices = list(score_matrix.columns)
        item_meta_df = build_item_metadata(metadata_by_model, all_indices)

        item_texts = [
            str(item_meta_df.at[idx, '_query_text'])
            if (idx in item_meta_df.index
                and '_query_text' in item_meta_df.columns
                and item_meta_df.at[idx, '_query_text'])
            else ''
            for idx in all_indices
        ]

        return score_matrix, item_texts, item_meta_df

    def _allocate_budget(
        self, strata: Dict[str, np.ndarray], target_size: int
    ) -> Dict[str, int]:
        """
        Allocate target_size across strata.

        When budget_fractions is None (default), fractions are derived from
        actual stratum sizes so the pruned set mirrors the difficulty
        distribution of the full benchmark.  A caller-supplied budget_fractions
        dict overrides this.  Rounding excess is resolved by trimming the
        largest stratum.
        """
        total = target_size
        floor = 1
        stratum_names = (_STRATUM_EASY, _STRATUM_MEDIUM, _STRATUM_HARD)

        if self.config.budget_fractions is not None:
            fracs = self.config.budget_fractions
        else:
            # Derive fractions from actual stratum sizes (proportional allocation)
            n_total_items = sum(len(strata.get(n, [])) for n in stratum_names)
            fracs = {
                n: len(strata.get(n, [])) / n_total_items if n_total_items > 0 else 0.0
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

        # Trim excess due to rounding or floor enforcement
        while sum(budgets.values()) > total:
            # Reduce whichever stratum has the most room above its floor
            candidates = [
                (name, b) for name, b in budgets.items()
                if b > floor and len(strata.get(name, [])) > 0
            ]
            if not candidates:
                break
            trim = max(candidates, key=lambda x: x[1])[0]
            budgets[trim] -= 1

        # If we're still under total (rare), top up medium stratum
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
        all_indices: List[int],
    ) -> np.ndarray:
        """Combine text embeddings and numeric metadata into a feature matrix."""
        parts: List[np.ndarray] = []

        # --- Text embeddings ---
        if item_texts:
            stratum_texts = [item_texts[p] for p in positions]
            has_content = any(t.strip() for t in stratum_texts)
            if has_content:
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
                    logger.warning(f'Embedding failed: {exc}; using metadata only.')

        # --- Numeric metadata features ---
        if item_metadata is not None and not item_metadata.empty:
            stratum_indices = [all_indices[p] for p in positions]
            meta_feats = self._numeric_metadata_features(
                item_metadata, stratum_indices
            )
            if meta_feats is not None:
                parts.append(meta_feats)

        if not parts:
            # Ultimate fallback: 1-D feature from item position index
            return np.arange(len(positions), dtype=float).reshape(-1, 1)

        return np.hstack(parts)

    def _numeric_metadata_features(
        self,
        meta_df: pd.DataFrame,
        indices: List[int],
    ) -> Optional[np.ndarray]:
        """Extract numeric features from per-item metadata."""
        has_tokens = 'input_tokens' in meta_df.columns
        has_dates = 'contest_date' in meta_df.columns
        if not has_tokens and not has_dates:
            return None

        n_feats = int(has_tokens) + int(has_dates)
        result = np.zeros((len(indices), n_feats), dtype=float)

        max_tokens = float(meta_df['input_tokens'].max() or 1) if has_tokens else 1.0
        if has_dates:
            _parsed_dates = []
            for _d in meta_df['contest_date'].dropna():
                try:
                    _p = str(_d).split('-')
                    _parsed_dates.append(
                        int(_p[0]) + ((int(_p[1]) - 1) / 12.0 if len(_p) > 1 else 0.0)
                    )
                except Exception:
                    pass
            date_start = min(_parsed_dates) if _parsed_dates else 2020.0
            date_span = max(max(_parsed_dates) - date_start, 1 / 12.0) if _parsed_dates else 1.0
        else:
            date_start = date_span = 0.0

        for row_i, idx in enumerate(indices):
            if idx not in meta_df.index:
                continue
            row = meta_df.loc[idx]
            col = 0
            if has_tokens:
                result[row_i, col] = float(row.get('input_tokens') or 0) / max_tokens
                col += 1
            if has_dates:
                result[row_i, col] = _date_recency(
                    str(row.get('contest_date', '')), date_start, date_span
                )

        return result


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _date_recency(date_str: str, start: float, span: float) -> float:
    """Map 'YYYY-MM-DD' to a float in [0, 1] using data-derived range."""
    try:
        parts = date_str.split('-')
        year = int(parts[0])
        month = int(parts[1]) if len(parts) > 1 else 1
        return max(0.0, min(1.0, (year + (month - 1) / 12.0 - start) / span))
    except Exception:
        return 0.0
