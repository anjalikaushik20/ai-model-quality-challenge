"""
Run the Spectral-IRT Stratified Pruner for a single benchmark.

Usage:
    python run_pruner.py <benchmark> <evals_dir> <output_dir> [target_size]

    benchmark  : live_code_bench_v5 | aa_lcr
    evals_dir  : directory containing predictions/ and reviews/ subdirectories
    output_dir : where to write selected_indices.json, pruning_report.json,
                 and validation_report.json
    target_size: optional integer (defaults per benchmark below)

Default target sizes are derived from stratum coverage analysis:
  the smallest N where every stratum's budget >= ceil(sqrt(stratum_size))
  clusters, cross-checked against the sensitivity sweep so both MAE < 0.10
  and KL < 0.05 are satisfied.

    live_code_bench_v5 : 72  (22.9% of 315; hard stratum needs 7 clusters,
                              quality threshold first met at n=72)
    aa_lcr             : 25  (25.0% of 100; hard stratum needs 5 clusters,
                              quality threshold first met at n=25)
"""

import json
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)-8s  %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)

MODELS = ['gpt-oss-120b', 'kimi-k2.5', 'minimax-m2.5']
KNOWN_BENCHMARKS = {'live_code_bench_v5', 'aa_lcr'}

# Target sizes derived from stratum coverage + sensitivity sweep (see module docstring)
DEFAULT_TARGETS = {
    'live_code_bench_v5': 72,
    'aa_lcr':             24,
}


def main():
    if len(sys.argv) < 4:
        print(__doc__)
        sys.exit(1)

    benchmark   = sys.argv[1]
    evals_dir   = sys.argv[2]
    output_dir  = sys.argv[3]

    if benchmark not in KNOWN_BENCHMARKS:
        logger.error(
            f'Unknown benchmark: {benchmark}. Choose from {sorted(KNOWN_BENCHMARKS)}'
        )
        sys.exit(1)

    # 0 or absent → use per-benchmark default derived from stratum coverage
    _arg = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    target_size = _arg if _arg > 0 else DEFAULT_TARGETS[benchmark]

    sys.path.insert(0, str(Path(__file__).parent))
    from pruner import PrunerConfig, SpectralIRTStratifiedPruner

    config = PrunerConfig(
        benchmark=benchmark,
        models=MODELS,
        evals_dir=evals_dir,
        target_size=target_size,
        cache_dir=str(Path(output_dir) / '.emb_cache'),
    )

    target_str = str(target_size)
    logger.info(f'Benchmark   : {benchmark}')
    logger.info(f'Evals dir   : {evals_dir}')
    logger.info(f'Output dir  : {output_dir}')
    logger.info(f'Target size : {target_str}')

    pruner = SpectralIRTStratifiedPruner(config)

    logger.info('── Fitting pruner ──')
    pruner.fit()

    logger.info('── Saving artefacts ──')
    pruner.save_artifacts(output_dir)

    logger.info('── Running validation ──')
    report = pruner.validate()

    report_path = Path(output_dir) / 'validation_report.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    logger.info(f'Validation report → {report_path}')

    v    = report.get('validation', {})
    sp   = v.get('score_preservation', {})
    rp   = v.get('ranking', {})
    gn   = v.get('go_nogo', {})
    lomo = v.get('lomo', {})
    kl   = v.get('kl_difficulty', {})

    print('\n── Results ─────────────────────────────────────────')
    print(f"  Full size        : {report['full_size']}")
    print(f"  Pruned size      : {report['pruned_size']}")
    print(f"  Compression      : {report['compression_ratio']:.1%}")

    print(f"\n  Score preservation  (simple mean)")
    print(f"    Mean |error|   : {sp.get('mean_absolute_error', 'N/A')}")
    print(f"    Max  |error|   : {sp.get('max_absolute_error', 'N/A')}")
    for m, d in sp.get('per_model', {}).items():
        print(
            f"    {m:22s}  full={d['full']:.3f}  "
            f"pruned={d['pruned']:.3f}  |err|={d['absolute_error']:.4f}"
        )

    print(f"\n  Ranking")
    print(f"    Spearman ρ     : {rp.get('spearman_rho', 'N/A')}")
    print(f"    Preserved      : {rp.get('ranking_preserved', 'N/A')}")
    if rp.get('rank_swaps'):
        print(f"    Swaps          : {rp['rank_swaps']}")

    print(f"\n  Go/No-Go  (threshold={gn.get('threshold', 'N/A')})")
    print(f"    Agreement      : {gn.get('agreement_rate', 'N/A'):.1%}")
    for m, d in gn.get('per_model', {}).items():
        sym = '✓' if d['agrees'] else '✗'
        print(
            f"    {sym} {m:22s}  full={d['full_decision']:6s}  "
            f"pruned={d['pruned_decision']}"
        )

    if lomo.get('folds'):
        print(f"\n  LOMO  (mean |error|={lomo.get('mean_absolute_error', 'N/A')})")
        for fold in lomo['folds']:
            if 'error' in fold:
                continue
            print(
                f"    {fold['held_out']:22s}  full={fold['full_score']:.3f}  "
                f"pruned={fold['pruned_score']:.3f}  |err|={fold['absolute_error']:.4f}"
            )

    if kl:
        print(f"\n  Difficulty KL     : {kl.get('kl_divergence', 'N/A')}")

    print('────────────────────────────────────────────────────\n')


if __name__ == '__main__':
    main()
