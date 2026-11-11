#!/usr/bin/env python3
"""One-command experiment: train once -> baseline -> sensitivity sweeps.

Official WikiText split mode (recommended for the thesis):
    python run_all.py --config configs/all_in_one_paper.json \
        --data-dir data/wikitext103

Backward-compatible single-corpus mode:
    python run_all.py --config configs/all_in_one_paper.json \
        --text corpus.txt

In official-split mode, train.txt / validation.txt / test.txt are used directly
and are NOT re-split.
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

from run_experiment import run_experiment
from run_sensitivity import run_sensitivity


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        '--config',
        required=True,
        help='All-in-one JSON containing training + photonic + sensitivity',
    )
    p.add_argument(
        '--text',
        default=None,
        help=(
            'Backward-compatible single UTF-8 corpus. The original 90/5/5 '
            'split is applied.'
        ),
    )
    p.add_argument(
        '--data-dir',
        default=None,
        help=(
            'Directory containing official train.txt, validation.txt and '
            'test.txt. No additional splitting is performed.'
        ),
    )
    return p.parse_args()


def main():
    args = parse_args()

    if args.text is not None and args.data_dir is not None:
        raise ValueError(
            'Use either --text or --data-dir, not both.'
        )

    config_path = Path(args.config)
    all_cfg = json.loads(
        config_path.read_text(encoding='utf-8')
    )
    if 'sensitivity' not in all_cfg:
        raise ValueError(
            'All-in-one config must contain a top-level "sensitivity" section.'
        )

    sensitivity_cfg = all_cfg['sensitivity']
    train_cfg = {
        k: v
        for k, v in all_cfg.items()
        if k != 'sensitivity'
    }

    root = Path(train_cfg.get('output_dir', 'runs'))
    run_dir = root / datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir.mkdir(parents=True, exist_ok=False)

    (run_dir / 'all_in_one_config_resolved.json').write_text(
        json.dumps(all_cfg, indent=2),
        encoding='utf-8',
    )

    data_mode = (
        'official_splits'
        if args.data_dir is not None
        else (
            'single_corpus_90_5_5'
            if args.text is not None
            else 'built_in_demo_90_5_5'
        )
    )

    print('=' * 78)
    print('PHASE 1/2  Train once + Digital/Realistic-Photonic baseline')
    print('Dataset mode:', data_mode)
    print('=' * 78)

    run_experiment(
        train_cfg,
        text_path=args.text,
        data_dir=args.data_dir,
        run_dir=run_dir,
    )

    print('\n' + '=' * 78)
    print('PHASE 2/2  Hardware + Nb sensitivity sweeps (NO retraining)')
    print('Dataset mode:', data_mode)
    print('=' * 78)

    sensitivity_dir = run_dir / 'sensitivity'
    run_sensitivity(
        run_dir,
        sensitivity_cfg,
        text_path=args.text,
        data_dir=args.data_dir,
        out_dir=sensitivity_dir,
    )

    summary = {
        'status': 'complete',
        'run_dir': str(run_dir),
        'baseline_dir': str(run_dir),
        'sensitivity_dir': str(sensitivity_dir),
        'trained_once_per_model': True,
        'sweeps_reuse_same_checkpoints': True,
        'dataset_mode': data_mode,
        'data_dir': args.data_dir,
        'text_path': args.text,
    }
    (run_dir / 'run_all_complete.json').write_text(
        json.dumps(summary, indent=2),
        encoding='utf-8',
    )

    print('\n' + '=' * 78)
    print('ALL DONE')
    print('=' * 78)
    print('Everything is under:', run_dir)
    print('Dataset record:', run_dir / 'dataset_info.json')
    print('Baseline scaling plots/CSVs:', run_dir)
    print('Sensitivity plots/CSVs:', sensitivity_dir)


if __name__ == '__main__':
    main()
