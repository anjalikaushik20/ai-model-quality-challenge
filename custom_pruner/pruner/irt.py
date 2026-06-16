"""
2PL-inspired IRT scoring for benchmark items.

With only 3 models, we cannot reliably fit a full 2PL model (too few
respondents for joint MLE/MCMC).  Instead we use explicit, non-fitted
estimates that mirror the 2PL parameterisation:

    difficulty_i     = 1 − mean(scores on item i)
    discrimination_i = point-biserial corr(scores_i, abilities), clipped to [0, 1]
    IRT_info_i       = discrimination_i² × P_i × (1 − P_i)
                       ≈ Fisher information at θ = b_i
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)


def estimate_abilities(score_matrix: np.ndarray) -> np.ndarray:
    """θ_m = mean score of model m across all items. Shape: (n_models,)"""
    return np.nanmean(score_matrix, axis=1)


def estimate_discrimination(
    score_matrix: np.ndarray,
    abilities: np.ndarray,
) -> np.ndarray:
    """
    Point-biserial correlation between each item's scores and model abilities,
    clipped to [0, 1].  NaN (constant item scores) → 0.

    Shape: (n_items,)
    """
    n_models, n_items = score_matrix.shape
    discriminations = np.zeros(n_items)

    for i in range(n_items):
        item_scores = score_matrix[:, i]
        valid = ~np.isnan(item_scores)
        if valid.sum() < 2 or item_scores[valid].std() < 1e-10:
            continue
        corr = np.corrcoef(item_scores[valid], abilities[valid])[0, 1]
        discriminations[i] = max(0.0, float(corr) if not np.isnan(corr) else 0.0)

    return discriminations


def compute_irt_info(
    discriminations: np.ndarray,
    pass_rates: np.ndarray,
) -> np.ndarray:
    """
    Fisher-information-inspired item quality score:
        I_i = a_i² × P_i × (1 − P_i)

    Peaks at P=0.5, zero at extremes. Shape: (n_items,)
    """
    return discriminations ** 2 * pass_rates * (1.0 - pass_rates)



def compute_pass_rates(score_matrix: np.ndarray) -> np.ndarray:
    """Mean score per item across models. Shape: (n_items,)."""
    return np.nanmean(score_matrix, axis=0)


def assign_strata_from_labels(difficulty_labels: np.ndarray) -> np.ndarray:
    """
    Map topic_difficulty strings to stratum names.

    Used when per-item difficulty labels are available directly from benchmark
    metadata (e.g. MMMU) rather than inferred from multi-model pass rates.

    Mapping:  'Easy' → 'easy' | 'Medium' → 'medium' | 'Hard' → 'hard'
    Unknown labels fall back to 'medium'.
    """
    _MAP = {'Easy': 'easy', 'Medium': 'medium', 'Hard': 'hard'}
    result = np.full(len(difficulty_labels), 'medium', dtype=object)
    for i, label in enumerate(difficulty_labels):
        result[i] = _MAP.get(str(label), 'medium')
    return result


def assign_strata(
    pass_rates: np.ndarray,
) -> np.ndarray:
    """
    Assign each item to a stratum based on pass rate.

    With 3 binary-scored models the only possible pass rates are
    0, 1/3, 2/3, 1.  We use inclusive boundaries:

        easy:   pass_rate >= 2/3   (2 or 3 models pass)
        hard:   pass_rate <= 1/3   (0 or 1 model passes)
        medium: everything else    (strictly between 1/3 and 2/3)

    Items at the 1/3 and 2/3 boundaries go to medium so that
    easy/hard strata retain only purely-agreed items (0 or 3 pass)
    as well as near-unanimous ones.  This keeps easy/hard strata
    non-empty while placing genuinely discriminating items in medium.

    Returns an array of strings: 'easy' | 'medium' | 'hard'
    """
    labels = np.full(len(pass_rates), 'medium', dtype=object)
    labels[pass_rates >= (2 / 3 - 1e-9)] = 'easy'
    labels[pass_rates <= (1 / 3 + 1e-9)] = 'hard'
    # Boundary items (exactly 1/3 or 2/3) match both rules above; force them
    # back to medium so easy/hard contain only unanimous items (pass_rate 0 or 1).
    at_low_boundary = np.isclose(pass_rates, 1 / 3)
    at_high_boundary = np.isclose(pass_rates, 2 / 3)
    labels[at_low_boundary | at_high_boundary] = 'medium'
    return labels
