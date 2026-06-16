"""
Sensitivity sweep: fit the pruner at multiple target sizes,
record MAE and KL-divergence, plot the curves, identify the elbow.
"""

import copy
import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Load validators by absolute path so this module works correctly whether it is
# imported as part of the package or loaded via importlib by Part B's stub
# (in which case relative imports would resolve to the wrong package).
from importlib.util import spec_from_file_location as _sfl, module_from_spec as _mfs

_v = _sfl('_sweep_vals', Path(__file__).parent / 'validators.py')
_vm = _mfs(_v)
_v.loader.exec_module(_vm)
_score_preservation = _vm.score_preservation
_kl_divergence_difficulty = _vm.kl_divergence_difficulty
del _v, _vm

_FRACTIONS = [
    0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20,
    0.23, 0.25, 0.28, 0.30, 0.35, 0.40, 0.50,
]


def run_sweep(
    pruner_class,
    config_template,
    score_matrix: pd.DataFrame,
    item_texts: Optional[List[str]] = None,
    item_metadata=None,
    fractions: Optional[List[float]] = None,
) -> List[dict]:
    """
    Fit pruner_class at each derived target size, record MAE and KL.

    Args:
        pruner_class: class with fit(score_matrix, item_texts, item_metadata) and select()
        config_template: PrunerConfig instance — copied and mutated per sweep point
        score_matrix: (n_models × n_items) DataFrame with pre-loaded data
        item_texts: list of item question texts (for embedding)
        item_metadata: per-item metadata DataFrame
        fractions: compression fractions to sweep (default: _FRACTIONS)

    Returns:
        List of dicts [{target_size, compression_ratio, mae, kl}] in size order.
    """
    if fractions is None:
        fractions = _FRACTIONS

    n_items = score_matrix.shape[1]
    sizes = sorted({max(3, round(n_items * f)) for f in fractions})

    full_scores = {
        m: float(np.nanmean(score_matrix.values[i]))
        for i, m in enumerate(score_matrix.index)
    }
    full_pass_rates = np.nanmean(score_matrix.values.astype(float), axis=0)
    all_cols = list(score_matrix.columns)

    curve: List[dict] = []
    for target in sizes:
        cfg = copy.deepcopy(config_template)
        cfg.target_size = target

        p = pruner_class(cfg)
        try:
            p.fit(
                score_matrix=score_matrix.copy(),
                item_texts=item_texts,
                item_metadata=item_metadata,
            )
            selected = p.select()
        except Exception as exc:
            logger.warning(f'sweep size={target} failed — {exc}')
            continue

        sel_pos = [all_cols.index(i) for i in selected if i in all_cols]
        if not sel_pos:
            continue

        pruned_scores = {
            m: float(np.nanmean(score_matrix.values[i, sel_pos]))
            for i, m in enumerate(score_matrix.index)
        }
        pruned_pr = np.nanmean(score_matrix.values.astype(float)[:, sel_pos], axis=0)

        sp = _score_preservation(full_scores, pruned_scores)
        kl = _kl_divergence_difficulty(full_pass_rates, pruned_pr)

        curve.append({
            'target_size': target,
            'compression_ratio': round(target / n_items, 4),
            'mae': sp['mean_absolute_error'],
            'kl': kl['kl_divergence'],
        })
        logger.info(
            f'  sweep  n={target:4d}  {target / n_items:.1%}'
            f'  MAE={sp["mean_absolute_error"]:.4f}  KL={kl["kl_divergence"]:.5f}'
        )

    return curve


def find_target(
    curve: List[dict],
    mae_tolerance: float = 0.50,
    kl_tolerance: float = 1.00,
    mae_floor: float = 0.005,
    kl_floor: float = 0.010,
) -> Optional[int]:
    """
    Smallest n where MAE and KL are within tolerance of their sweep-minimum values.

    Ceilings are computed as:
        mae_ceil = max(min_mae × (1 + mae_tolerance),  mae_floor)
        kl_ceil  = max(min_kl  × (1 + kl_tolerance),  kl_floor)

    The floors prevent near-zero minimums (e.g. MMMU's binary KL) from setting
    an impossibly tight ceiling and forcing the largest possible n.  With floors:
      - A stable benchmark still requires MAE ≤ 0.5 pp and KL ≤ 0.01, so it
        won't pick its worst-quality (smallest-n) point just because the absolute
        scale is tiny everywhere.
      - A noisy benchmark still uses the relative ceiling, which adapts to its
        own scale.

    mae_tolerance=0.50  → accept up to 1.5 × sweep-minimum MAE
    kl_tolerance=1.00   → accept up to 2.0 × sweep-minimum KL
    mae_floor=0.005     → never require MAE < 0.5 pp (absolute)
    kl_floor=0.010      → never require KL  < 0.01  (absolute)
    """
    if not curve:
        return None
    if len(curve) == 1:
        return curve[0]['target_size']

    min_mae = min(p['mae'] for p in curve)
    min_kl  = min(p['kl']  for p in curve)

    mae_ceil = max(min_mae * (1 + mae_tolerance), mae_floor)
    kl_ceil  = max(min_kl  * (1 + kl_tolerance),  kl_floor)

    for p in curve:
        if p['mae'] <= mae_ceil and p['kl'] <= kl_ceil:
            return p['target_size']

    # Unreachable in practice — minimum point always satisfies its own ceiling.
    best = min(curve, key=lambda p: p['mae'] / (min_mae + 1e-12) + p['kl'] / (min_kl + 1e-12))
    logger.warning(f'No point met relative thresholds; using best: n={best["target_size"]}')
    return best['target_size']


def plot_sweep(
    curve: List[dict],
    output_dir: str,
    benchmark: str,
    target_size: Optional[int] = None,
) -> Optional[str]:
    """
    Save a two-panel MAE / KL sweep figure to output_dir/sensitivity_sweep_{benchmark}.png.
    Returns the saved path, or None if matplotlib is unavailable.
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker
    except ImportError:
        logger.warning('matplotlib unavailable — skipping sweep plot')
        return None

    sizes  = [p['target_size']       for p in curve]
    maes   = [p['mae']               for p in curve]
    kls    = [p['kl']                for p in curve]
    ratios = [p['compression_ratio'] for p in curve]
    # Recover full benchmark size from the last (largest) sweep point
    n_full = max(1, round(sizes[-1] / ratios[-1])) if ratios[-1] > 0 else sizes[-1]

    fig, (ax_mae, ax_kl) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    fig.suptitle(f'{benchmark} — Sensitivity Sweep', fontsize=13, fontweight='bold')

    # ── MAE panel ──────────────────────────────────────────────────────────
    ax_mae.plot(sizes, maes, 'o-', color='steelblue', lw=2, ms=5)
    ax_mae.set_ylabel('Mean Absolute Error', fontsize=11)
    ax_mae.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.3f'))
    ax_mae.grid(True, alpha=0.3)

    # ── KL panel ───────────────────────────────────────────────────────────
    ax_kl.plot(sizes, kls, 's-', color='darkorange', lw=2, ms=5)
    ax_kl.set_ylabel('KL Divergence (difficulty)', fontsize=11)
    ax_kl.set_xlabel('Target size (items)', fontsize=11)
    ax_kl.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.4f'))
    ax_kl.grid(True, alpha=0.3)

    # ── Selected target marker ─────────────────────────────────────────────
    if target_size is not None:
        target_ratio = target_size / n_full
        for ax in (ax_mae, ax_kl):
            ax.axvline(
                target_size, color='green', ls='--', lw=1.8,
                label=f'Target  n={target_size}  ({target_ratio:.0%})',
            )
        ax_mae.legend(fontsize=9, loc='upper right')
        ax_kl.legend(fontsize=9, loc='upper right')

    # ── Secondary x-axis: compression ratio ────────────────────────────────
    ax_top = ax_mae.twiny()
    ax_top.set_xlim(ax_mae.get_xlim())
    tick_sz = sizes[::max(1, len(sizes) // 6)]
    ax_top.set_xticks(tick_sz)
    ax_top.set_xticklabels([f'{s / n_full:.0%}' for s in tick_sz], fontsize=8)
    ax_top.set_xlabel('Compression ratio (% retained)', fontsize=9)

    plt.tight_layout()

    out = Path(output_dir) / f'sensitivity_sweep_{benchmark}.png'
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f'Sweep plot → {out}')
    return str(out)
