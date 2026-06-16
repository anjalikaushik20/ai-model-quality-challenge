#!/usr/bin/env python3
"""
Generate stratum-coverage plots for LCB, AA-LCR, and MMMU.

Each plot shows, for a range of candidate target sizes n:
  - How many items are allocated to each difficulty stratum (easy/medium/hard)
  - The per-stratum requirement: ceil(sqrt(stratum_size)) spectral clusters
  - The stratum-coverage minimum n (first n where all requirements are met)
  - The chosen target n

Output:
  output/lcb/stratum_coverage_live_code_bench_v5.png
  output/aalcr/stratum_coverage_aa_lcr.png
  part_b/output/stratum_coverage_mmmu.png
"""

import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── Replicate the pruner's _allocate_budget logic ─────────────────────────────

def allocate_budget(strata_sizes, total):
    """
    Proportional budget allocation matching pruner._allocate_budget.
    strata_sizes: dict {name: size}
    Returns dict {name: budget}
    """
    names   = list(strata_sizes.keys())
    n_total = sum(strata_sizes.values())
    fracs   = {n: strata_sizes[n] / n_total for n in names}
    floor   = 1

    budgets = {}
    for name in names:
        available = strata_sizes[name]
        if available == 0:
            budgets[name] = 0
            continue
        raw = round(total * fracs[name])
        budgets[name] = max(floor, min(raw, available))

    # Trim rounding excess from the largest stratum
    while sum(budgets.values()) > total:
        candidates = [(n, b) for n, b in budgets.items() if b > floor and strata_sizes[n] > 0]
        if not candidates:
            break
        trim = max(candidates, key=lambda x: x[1])[0]
        budgets[trim] -= 1

    # Top up medium if under (rare)
    while sum(budgets.values()) < total:
        med = 'medium'
        if budgets[med] < strata_sizes[med]:
            budgets[med] += 1
        else:
            break

    return budgets


def compute_coverage_minimum(strata_sizes, requirements):
    """
    Analytical minimum n: ceil(req * N_total / S_hard).
    The hard stratum is always binding (smallest, slowest to accumulate budget).
    """
    n_total = sum(strata_sizes.values())
    return max(
        math.ceil(requirements[s] * n_total / strata_sizes[s])
        for s in strata_sizes
    )


def plot_coverage(strata_sizes, chosen_target, out_path, title, n_range=None):
    """
    Two-panel figure (budget curves + per-stratum requirement headroom).
    """
    if n_range is None:
        n_total = sum(strata_sizes.values())
        n_range = range(3, int(n_total * 0.60) + 1)

    requirements = {s: math.ceil(math.sqrt(sz)) for s, sz in strata_sizes.items()}
    coverage_min = compute_coverage_minimum(strata_sizes, requirements)

    ns = list(n_range)
    budgets_by_n = [allocate_budget(strata_sizes, n) for n in ns]

    colors = {'easy': '#4c9ed9', 'medium': '#f5a623', 'hard': '#e05252'}
    labels = {'easy': 'Easy', 'medium': 'Medium', 'hard': 'Hard'}

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.suptitle(title, fontsize=13, fontweight='bold')

    # ── Budget curves ──────────────────────────────────────────────────────────
    for stratum in ('easy', 'medium', 'hard'):
        vals = [b[stratum] for b in budgets_by_n]
        req  = requirements[stratum]
        col  = colors[stratum]

        ax.plot(ns, vals, '-', color=col, lw=2,
                label=f'{labels[stratum]} (need ≥ {req})')

        # Gray threshold line at requirement
        ax.axhline(req, color='gray', ls=':', lw=1.2, alpha=0.8)

        # Annotate with stratum name at right edge
        ax.annotate(
            labels[stratum],
            xy=(ns[-1], req),
            xytext=(4, 0),
            textcoords='offset points',
            va='center',
            fontsize=8,
            color='gray',
        )

    # ── Coverage-minimum marker ────────────────────────────────────────────────
    if coverage_min is not None:
        ax.axvline(coverage_min, color='gray', ls=':', lw=1.6,
                   label=f'Min coverage  n={coverage_min}')

    # ── Chosen-target marker ───────────────────────────────────────────────────
    n_total = sum(strata_sizes.values())
    pct     = chosen_target / n_total
    ax.axvline(chosen_target, color='green', ls='--', lw=2.0,
               label=f'Chosen target  n={chosen_target}  ({pct:.0%})')

    ax.set_xlabel('Target size n (items selected)', fontsize=11)
    ax.set_ylabel('Items allocated to stratum', fontsize=11)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True, nbins=10))
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc='upper left')

    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved → {out_path}')


# ── Benchmark configs ──────────────────────────────────────────────────────────

BASE = Path(__file__).parent

BENCHMARKS = [
    {
        'title':   'LCB — Stratum Coverage',
        'strata':  {'easy': 158, 'medium': 111, 'hard': 46},
        'chosen':  72,
        'out':     BASE / 'output/lcb/stratum_coverage_live_code_bench_v5.png',
        'n_range': range(3, 200),
    },
    {
        'title':   'AA-LCR — Stratum Coverage',
        'strata':  {'easy': 36, 'medium': 43, 'hard': 21},
        'chosen':  24,
        'out':     BASE / 'output/aalcr/stratum_coverage_aa_lcr.png',
        'n_range': range(3, 70),
    },
    {
        'title':   'MMMU — Stratum Coverage',
        'strata':  {'easy': 108, 'medium': 173, 'hard': 79},
        'chosen':  42,
        'out':     BASE / 'part_b/output/stratum_coverage_mmmu.png',
        'n_range': range(3, 200),
    },
]

for cfg in BENCHMARKS:
    plot_coverage(
        strata_sizes=cfg['strata'],
        chosen_target=cfg['chosen'],
        out_path=cfg['out'],
        title=cfg['title'],
        n_range=cfg['n_range'],
    )
