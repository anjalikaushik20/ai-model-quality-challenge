"""
Run the MMMU benchmark pruner (Part B).

Usage:
    python run_pruner.py <evals_dir> <output_dir> [target_size]

    evals_dir  : directory containing predictions/ and reviews/ subdirectories
                 (e.g. /data/.../Evals/MMMU)
    output_dir : where to write selected_indices.json, pruning_report.json,
                 and validation_report.json
    target_size: optional integer (default: 60)
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

MODEL = 'glm-4.5v-fp8'

# Target derived from stratum coverage: hard stratum (79 items) needs
# ceil(sqrt(79)) = 9 clusters, binding N = ceil(9 * 360/79) = 42 (11.7%).
# Sensitivity sweep confirms quality is acceptable from n=30 onwards.
DEFAULT_TARGET = 42


def main() -> None:
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    evals_dir   = sys.argv[1]
    output_dir  = sys.argv[2]
    target_size = int(sys.argv[3]) if len(sys.argv) > 3 else DEFAULT_TARGET

    sys.path.insert(0, str(Path(__file__).parent))
    from pruner import PrunerConfig, SpectralStratifiedPruner

    config = PrunerConfig(
        benchmark='mmmu',
        models=[MODEL],
        evals_dir=evals_dir,
        target_size=target_size,
        cache_dir=str(Path(output_dir) / '.emb_cache'),
    )

    logger.info(f'Benchmark   : mmmu')
    logger.info(f'Model       : {MODEL}')
    logger.info(f'Evals dir   : {evals_dir}')
    logger.info(f'Output dir  : {output_dir}')
    logger.info(f'Target size : {target_size}')

    pruner = SpectralStratifiedPruner(config)

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

    v  = report.get('validation', {})
    sp = v.get('score_preservation', {})
    gn = v.get('go_nogo', {})
    kl = v.get('kl_difficulty', {})

    print('\n── Results ─────────────────────────────────────────')
    print(f"  Full size        : {report['full_size']}")
    print(f"  Pruned size      : {report['pruned_size']}")
    print(f"  Compression      : {report['compression_ratio']:.1%}")

    print(f"\n  Stratum sizes (full)  : {report.get('stratum_counts', {})}")
    print(f"  Stratum budgets       : {report.get('stratum_budgets', {})}")

    print(f"\n  Score preservation")
    print(f"    Mean |error|   : {sp.get('mean_absolute_error', 'N/A')}")
    print(f"    Max  |error|   : {sp.get('max_absolute_error', 'N/A')}")
    for m, d in sp.get('per_model', {}).items():
        print(
            f"    {m:26s}  full={d['full']:.3f}  "
            f"pruned={d['pruned']:.3f}  |err|={d['absolute_error']:.4f}"
        )

    print(f"\n  Go/No-Go  (threshold={gn.get('threshold', 'N/A')})")
    print(f"    Agreement      : {gn.get('agreement_rate', 'N/A'):.1%}")
    for m, d in gn.get('per_model', {}).items():
        sym = '✓' if d['agrees'] else '✗'
        print(
            f"    {sym} {m:26s}  full={d['full_decision']:6s}  "
            f"pruned={d['pruned_decision']}"
        )

    if kl:
        print(f"\n  Difficulty KL     : {kl.get('kl_divergence', 'N/A')}")

    print('────────────────────────────────────────────────────\n')


if __name__ == '__main__':
    main()
