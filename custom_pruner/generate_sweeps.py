#!/usr/bin/env python3
"""
Generate sensitivity sweep plots for LCB, AA-LCR, and MMMU.

LCB + AA-LCR : sweeps the pruner across 14 target sizes via run_sweep().
MMMU          : builds the curve from mmmu_t30 / mmmu_t60 / mmmu_t90 / mmmu_t120
                validation reports (runs t30 first if the folder does not exist).

Output plots:
  output/lcb/sensitivity_sweep_live_code_bench_v5.png
  output/aalcr/sensitivity_sweep_aa_lcr.png
  part_b/output/sensitivity_sweep_mmmu.png
"""

import json
import logging
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)-8s  %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)

BASE = Path(__file__).parent
sys.path.insert(0, str(BASE))

from pruner import PrunerConfig, SpectralIRTStratifiedPruner
from pruner.sweep import run_sweep

MODELS      = ['gpt-oss-120b', 'kimi-k2.5', 'minimax-m2.5']
PART_A_EVALS = '/data/akaush39/ai-model-quality-challenge/Evals/Part 1'
MMMU_EVALS   = '/data/akaush39/ai-model-quality-challenge/Evals/MMMU'


def save_plot(curve, out_path, title, n_full, chosen_target,
              mae_ref=0.10, kl_ref=0.05):
    """Two-panel MAE / KL sweep plot with threshold lines and chosen-target marker."""
    sizes  = [p['target_size']        for p in curve]
    maes   = [p['mae']                for p in curve]
    kls    = [p['kl']                 for p in curve]
    ratios = [p['compression_ratio']  for p in curve]

    fig, (ax_mae, ax_kl) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    fig.suptitle(title, fontsize=13, fontweight='bold')

    # ── MAE panel ──────────────────────────────────────────────────────────
    ax_mae.plot(sizes, maes, 'o-', color='steelblue', lw=2, ms=5, label='MAE')
    ax_mae.axhline(mae_ref, color='gray', ls=':', lw=1.4,
                   label=f'Quality threshold  ({mae_ref:.2f})')
    ax_mae.set_ylabel('Mean Absolute Error', fontsize=11)
    ax_mae.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.3f'))
    ax_mae.grid(True, alpha=0.3)

    # ── KL panel ───────────────────────────────────────────────────────────
    ax_kl.plot(sizes, kls, 's-', color='darkorange', lw=2, ms=5,
               label='KL divergence')
    ax_kl.axhline(kl_ref, color='gray', ls=':', lw=1.4,
                  label=f'Quality threshold  ({kl_ref:.2f})')
    ax_kl.set_ylabel('KL Divergence (difficulty)', fontsize=11)
    ax_kl.set_xlabel('Target size (items)', fontsize=11)
    ax_kl.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.4f'))
    ax_kl.grid(True, alpha=0.3)

    # ── Chosen-target marker ────────────────────────────────────────────────
    ratio_pct = chosen_target / n_full
    for ax in (ax_mae, ax_kl):
        ax.axvline(chosen_target, color='green', ls='--', lw=1.8,
                   label=f'Chosen target  n={chosen_target}  ({ratio_pct:.0%})')

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
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f'Saved → {out_path}')


# ── Part A: LCB and AA-LCR ─────────────────────────────────────────────────

for benchmark, out_dir, chosen in [
    ('live_code_bench_v5', BASE / 'output/lcb',   72),
    ('aa_lcr',             BASE / 'output/aalcr', 25),
]:
    logger.info(f'══ Sweeping {benchmark} ══')
    config = PrunerConfig(
        benchmark=benchmark,
        models=MODELS,
        evals_dir=PART_A_EVALS,
        target_size=1,          # overridden inside run_sweep per iteration
        target_fraction=0.25,
        cache_dir=str(out_dir / '.emb_cache'),
    )
    sm, texts, meta = SpectralIRTStratifiedPruner(config)._load_data()
    curve = run_sweep(SpectralIRTStratifiedPruner, config, sm, texts, meta)
    save_plot(
        curve,
        out_dir / f'sensitivity_sweep_{benchmark}.png',
        f'{benchmark} — Sensitivity Sweep',
        sm.shape[1],
        chosen,
    )

# ── Part B: MMMU ────────────────────────────────────────────────────────────

mmmu_base = BASE / 'part_b/output'

# Run t30 if it hasn't been produced yet
t30_report = mmmu_base / 'mmmu_t30' / 'validation_report.json'
if not t30_report.exists():
    logger.info('Running MMMU pruner at target=30 ...')
    subprocess.run(
        [sys.executable,
         str(BASE / 'part_b/run_pruner.py'),
         MMMU_EVALS,
         str(mmmu_base / 'mmmu_t30'),
         '30'],
        check=True,
    )

# Build curve from the four sweep folders
mmmu_curve = []
for t in [30, 60, 90, 120]:
    rpath = mmmu_base / f'mmmu_t{t}' / 'validation_report.json'
    if not rpath.exists():
        logger.warning(f'Missing {rpath} — skipping point t={t}')
        continue
    r = json.loads(rpath.read_text())
    v = r['validation']
    mmmu_curve.append({
        'target_size':       r['pruned_size'],
        'compression_ratio': r['compression_ratio'],
        'mae': v['score_preservation']['mean_absolute_error'],
        'kl':  v['kl_difficulty']['kl_divergence'],
    })

logger.info(f'MMMU curve points: {mmmu_curve}')
save_plot(
    mmmu_curve,
    mmmu_base / 'sensitivity_sweep_mmmu.png',
    'mmmu — Sensitivity Sweep',
    360,
    42,
)
