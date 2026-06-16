"""
Validation suite for pruned benchmark subsets.

Metrics
-------
score_preservation       — per-model MAE (simple mean, like-for-like)
ranking_preservation     — Spearman's ρ between full/pruned rankings
go_nogo_agreement        — does the pruned subset produce the same GO/NO-GO decision?
kl_divergence_difficulty — KL divergence between difficulty distributions
lomo_validation          — leave-one-model-out generalisation test
"""

import copy
import logging
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Score preservation
# ---------------------------------------------------------------------------

def score_preservation(
    full_scores: Dict[str, float],
    pruned_scores: Dict[str, float],
) -> dict:
    """Per-model and aggregate absolute error between full and pruned scores."""
    models = list(full_scores)
    errors = {m: abs(full_scores[m] - pruned_scores[m]) for m in models}
    return {
        'per_model': {
            m: {
                'full': round(full_scores[m], 4),
                'pruned': round(pruned_scores[m], 4),
                'absolute_error': round(errors[m], 4),
            }
            for m in models
        },
        'mean_absolute_error': round(float(np.mean(list(errors.values()))), 4),
        'max_absolute_error': round(float(np.max(list(errors.values()))), 4),
    }



# ---------------------------------------------------------------------------
# Ranking preservation (Kendall's τ + Spearman's ρ)
# ---------------------------------------------------------------------------

def ranking_preservation(
    full_scores: Dict[str, float],
    pruned_scores: Dict[str, float],
) -> dict:
    """Spearman's ρ between full and pruned model rankings, plus swap detection."""
    models = list(full_scores)
    full_vals = [full_scores[m] for m in models]
    pruned_vals = [pruned_scores[m] for m in models]

    if len(models) < 2:
        rho, rho_p = 1.0, 1.0
    else:
        rho_stat, rho_p = spearmanr(full_vals, pruned_vals)
        rho = float(rho_stat) if not np.isnan(rho_stat) else None

    swaps = []
    for i, m1 in enumerate(models):
        for m2 in models[i + 1:]:
            if (full_scores[m1] > full_scores[m2]) != (pruned_scores[m1] > pruned_scores[m2]):
                swaps.append({
                    'model_a': m1,
                    'model_b': m2,
                    'full_delta': round(full_scores[m1] - full_scores[m2], 4),
                    'pruned_delta': round(pruned_scores[m1] - pruned_scores[m2], 4),
                })

    return {
        'spearman_rho': rho,
        'spearman_p': round(float(rho_p), 4) if rho_p is not None else None,
        'ranking_preserved': len(swaps) == 0,
        'rank_swaps': swaps,
        'full_ranking': sorted(models, key=lambda m: full_scores[m], reverse=True),
        'pruned_ranking': sorted(models, key=lambda m: pruned_scores[m], reverse=True),
    }


# ---------------------------------------------------------------------------
# Go / No-Go agreement
# ---------------------------------------------------------------------------

def go_nogo_agreement(
    full_scores: Dict[str, float],
    pruned_scores: Dict[str, float],
    threshold: float,
) -> dict:
    """
    Check whether the pruned benchmark gives the same GO/NO-GO decision
    as the full benchmark for every model.
    """
    models = list(full_scores)
    full_dec = {m: full_scores[m] >= threshold for m in models}
    pruned_dec = {m: pruned_scores[m] >= threshold for m in models}
    agree = {m: full_dec[m] == pruned_dec[m] for m in models}

    return {
        'threshold': round(threshold, 4),
        'per_model': {
            m: {
                'full_decision': 'GO' if full_dec[m] else 'NO-GO',
                'pruned_decision': 'GO' if pruned_dec[m] else 'NO-GO',
                'agrees': agree[m],
            }
            for m in models
        },
        'agreement_rate': round(float(np.mean(list(agree.values()))), 4),
        'all_agree': all(agree.values()),
    }


# ---------------------------------------------------------------------------
# KL divergence on difficulty distributions
# ---------------------------------------------------------------------------

def kl_divergence_difficulty(
    pass_rates_full: np.ndarray,
    pass_rates_pruned: np.ndarray,
    n_bins: int = 5,
) -> dict:
    """
    KL(full ‖ pruned) on binned pass-rate distributions.

    Low KL means the pruned set preserves the difficulty structure of the
    full benchmark.
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    eps = 1e-9

    full_hist, _ = np.histogram(pass_rates_full, bins=bins)
    pruned_hist, _ = np.histogram(pass_rates_pruned, bins=bins)

    full_dist = (full_hist + eps) / (full_hist.sum() + eps * n_bins)
    pruned_dist = (pruned_hist + eps) / (pruned_hist.sum() + eps * n_bins)

    kl = float(np.sum(full_dist * np.log(full_dist / pruned_dist)))

    return {
        'kl_divergence': round(kl, 5),
        'full_histogram': full_hist.tolist(),
        'pruned_histogram': pruned_hist.tolist(),
        'bin_edges': [round(b, 2) for b in bins.tolist()],
    }


# ---------------------------------------------------------------------------
# Leave-one-model-out validation
# ---------------------------------------------------------------------------

def lomo_validation(
    score_matrix: np.ndarray,
    model_names: List[str],
    all_indices: List[int],
    config,
) -> dict:
    """
    Leave-one-model-out (LOMO) validation.

    For each held-out model:
      1. Fit a fresh pruner using the remaining N-1 models.
      2. Score the held-out model on full vs pruned subset (simple mean).

    Tests whether the selected items generalise to a model the pruner
    has never seen — directly addressing the overfitting-to-3-models concern.
    """
    # Lazy import — avoids circular dependency (this module is imported by pruner.py).
    from .pruner import SpectralIRTStratifiedPruner

    n_models = score_matrix.shape[0]
    folds = []

    for hold_idx in range(n_models):
        held_model = model_names[hold_idx]
        train_idx = [i for i in range(n_models) if i != hold_idx]
        train_names = [model_names[i] for i in train_idx]

        sub_df = pd.DataFrame(
            score_matrix[train_idx, :],
            index=train_names,
            columns=all_indices,
        )

        lomo_cfg = copy.deepcopy(config)
        lomo_cfg.models = train_names

        pruner = SpectralIRTStratifiedPruner(lomo_cfg)
        try:
            pruner.fit(score_matrix=sub_df)
            selected = pruner.select()
        except Exception as exc:
            logger.warning(f'LOMO fold {held_model}: fit failed — {exc}')
            folds.append({'held_out': held_model, 'error': str(exc)})
            continue

        held_scores = score_matrix[hold_idx, :]
        full_score = float(np.nanmean(held_scores))

        sel_pos = [all_indices.index(i) for i in selected if i in all_indices]

        # Use simple mean — LOMO tests representativeness, not discrimination
        if sel_pos:
            pruned_score = float(np.nanmean(held_scores[sel_pos]))
        else:
            pruned_score = float('nan')

        folds.append({
            'held_out': held_model,
            'full_score': round(full_score, 4),
            'pruned_score': round(pruned_score, 4),
            'absolute_error': round(abs(full_score - pruned_score), 4),
        })

    errors = [f['absolute_error'] for f in folds if 'absolute_error' in f]
    return {
        'folds': folds,
        'mean_absolute_error': round(float(np.mean(errors)), 4) if errors else None,
        'max_absolute_error': round(float(np.max(errors)), 4) if errors else None,
        'note': (
            'N=3 LOMO is suggestive of generalisation; '
            'validate on public leaderboard scores for stronger evidence.'
        ),
    }
