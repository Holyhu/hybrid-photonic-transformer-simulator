#!/usr/bin/env python3
import argparse
import csv
import json
import math
import platform
import random
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from analysis.scaling import fit_power_law
from data.text import (
    ByteTokenizer,
    dataset_info,
    load_data_splits,
    sequential_batches,
)
from train_eval import train_digital, evaluate, calibrate
from models.transformer import (
    TinyGPT,
    ModelSpec,
    total_parameter_count,
    non_embedding_parameter_count,
)
from photonic.config import PhotonicConfig
from photonic.core import PhotonicLinearEngine
from photonic.stats import PHOTO_STATS


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument(
        '--text',
        default=None,
        help=(
            'Backward-compatible single UTF-8 corpus. The program applies '
            'the original 90/5/5 split.'
        ),
    )
    p.add_argument(
        '--data-dir',
        default=None,
        help=(
            'Directory containing official train.txt, validation.txt and '
            'test.txt. These splits are used directly without re-splitting.'
        ),
    )
    return p.parse_args()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(name):
    return torch.device(
        'cuda'
        if name != 'cpu' and torch.cuda.is_available()
        else 'cpu'
    )


@torch.no_grad()
def evaluate_with_logits(
    model,
    data,
    batch_size,
    block_size,
    device,
    max_batches,
    engine=None,
    calibration=None,
    photonic_lm_head=True,
):
    model.eval()
    losses = []
    logits_all = []

    for x, y in sequential_batches(
        data,
        batch_size,
        block_size,
        device,
        max_batches,
    ):
        logits, loss = model(
            x,
            y,
            engine=engine,
            calibration=calibration,
            photonic_lm_head=photonic_lm_head,
        )
        losses.append(float(loss.detach().cpu()))
        logits_all.append(logits.detach().float().cpu())

    if not losses:
        raise RuntimeError('No evaluation batches were produced.')

    loss = sum(losses) / len(losses)
    return (
        loss,
        math.exp(min(loss, 20.0)),
        torch.cat(logits_all, dim=0),
    )


def calculate_nb_system(photonic_logits, digital_logits):
    ideal = digital_logits.double()
    actual = photonic_logits.double()
    err = actual - ideal

    rmse = float(
        torch.sqrt(torch.mean(err.square())).item()
    )
    ideal_max = float(ideal.max().item())
    ideal_min = float(ideal.min().item())
    dynamic_range = ideal_max - ideal_min

    eps = np.finfo(np.float64).tiny
    nb = math.log2(
        max(dynamic_range, eps) / max(rmse, eps)
    )

    return {
        'system_ideal_max': ideal_max,
        'system_ideal_min': ideal_min,
        'system_dynamic_range': dynamic_range,
        'system_error_rms': rmse,
        'Nb_system_rms': nb,
    }


def write_csv(path, rows):
    if not rows:
        return

    fieldnames = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    with open(path, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def env_info():
    return {
        'python': platform.python_version(),
        'torch': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'cuda_version': torch.version.cuda,
        'gpu': (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else None
        ),
    }


def plot_all(rows, out):
    N = np.array(
        [r['N_nonembedding'] for r in rows],
        float,
    )
    digital = np.array(
        [r['digital_loss'] for r in rows],
        float,
    )
    photonic = np.array(
        [r['realistic_photonic_loss'] for r in rows],
        float,
    )

    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111)
    ax.loglog(N, digital, 'o-', label='Digital')
    ax.loglog(
        N,
        photonic,
        '^-',
        label='Realistic photonic',
    )
    ax.set_xlabel('Non-embedding parameter count N')
    ax.set_ylabel('Test cross-entropy loss')
    ax.grid(True, which='both', alpha=.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / 'loss_vs_N.png', dpi=220)
    plt.close(fig)

    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111)
    ax.semilogx(N, photonic - digital, 'o-')
    ax.axhline(0, lw=1)
    ax.set_xlabel('N')
    ax.set_ylabel('Photonic loss - digital loss')
    ax.grid(True, which='both', alpha=.25)
    fig.tight_layout()
    fig.savefig(out / 'delta_loss_vs_N.png', dpi=220)
    plt.close(fig)

    finite_rows = [
        r for r in rows
        if np.isfinite(float(r['Nb_system_rms']))
    ]
    if finite_rows:
        fig = plt.figure(figsize=(7, 5))
        ax = fig.add_subplot(111)
        ax.semilogx(
            [r['N_nonembedding'] for r in finite_rows],
            [r['Nb_system_rms'] for r in finite_rows],
            'o-',
        )
        ax.set_xlabel('Non-embedding parameter count N')
        ax.set_ylabel(
            r'System effective precision $N_{b,\mathrm{sys}}$ (bits)'
        )
        ax.grid(True, which='both', alpha=.25)
        fig.tight_layout()
        fig.savefig(out / 'Nb_system_vs_N.png', dpi=220)
        plt.close(fig)


def _load_config(config):
    if isinstance(config, (str, Path)):
        return json.loads(
            Path(config).read_text(encoding='utf-8')
        )
    return dict(config)


def run_experiment(
    config,
    text_path=None,
    data_dir=None,
    run_dir=None,
):
    """Train all model sizes and evaluate digital/photonic baselines.

    ``data_dir`` enables official-split mode and expects:
        train.txt / validation.txt / test.txt

    ``text_path`` keeps the original one-file 90/5/5 behavior.
    """
    cfg = _load_config(config)
    seed_all(cfg['seed'])
    device = choose_device(cfg.get('device', 'auto'))

    if run_dir is None:
        out = (
            Path(cfg.get('output_dir', 'runs'))
            / datetime.now().strftime('%Y%m%d_%H%M%S')
        )
        out.mkdir(parents=True, exist_ok=True)
    else:
        out = Path(run_dir)
        out.mkdir(parents=True, exist_ok=True)

    (out / 'config_resolved.json').write_text(
        json.dumps(cfg, indent=2),
        encoding='utf-8',
    )
    (out / 'environment.json').write_text(
        json.dumps(env_info(), indent=2),
        encoding='utf-8',
    )

    train, val, test = load_data_splits(
        text_path=text_path,
        data_dir=data_dir,
    )
    ds_info = dataset_info(
        train,
        val,
        test,
        text_path=text_path,
        data_dir=data_dir,
    )
    (out / 'dataset_info.json').write_text(
        json.dumps(ds_info, indent=2),
        encoding='utf-8',
    )

    print('\nDataset mode:', ds_info['mode'])
    print('Training dataset size D:', ds_info['train_tokens_D'], 'byte tokens')
    print('Validation tokens:', ds_info['validation_tokens'])
    print('Test tokens:', ds_info['test_tokens'])

    pcfg = PhotonicConfig(**cfg['photonic'])
    specs = [
        ModelSpec(**s)
        for s in cfg['model_specs']
    ]

    rows = []
    all_layers = []

    for mi, spec in enumerate(specs):
        print('\n' + '=' * 70)
        print(spec)
        seed_all(cfg['seed'] + 1000 * mi)

        model = TinyGPT(
            ByteTokenizer.vocab_size,
            cfg['block_size'],
            spec,
        ).to(device)

        best = train_digital(
            model,
            train,
            val,
            cfg,
            device,
        )
        n_total = total_parameter_count(model)
        n_nonembedding = non_embedding_parameter_count(model)

        digital_loss, digital_ppl, digital_logits = evaluate_with_logits(
            model,
            test,
            cfg['batch_size'],
            cfg['block_size'],
            device,
            cfg['test_batches'],
        )

        calibration = calibrate(
            model,
            val,
            cfg,
            device,
            pcfg.input_percentile,
        )

        PHOTO_STATS.reset()
        realistic_engine = PhotonicLinearEngine(
            pcfg,
            'realistic',
        )
        (
            photonic_loss,
            photonic_ppl,
            photonic_logits,
        ) = evaluate_with_logits(
            model,
            test,
            cfg['batch_size'],
            cfg['block_size'],
            device,
            cfg['test_batches'],
            realistic_engine,
            calibration,
            pcfg.photonic_lm_head,
        )

        nb_metrics = calculate_nb_system(
            photonic_logits,
            digital_logits,
        )

        layer_stats = PHOTO_STATS.rows()
        write_csv(
            out / f'layer_stats_{spec.name}_realistic.csv',
            layer_stats,
        )
        for row in layer_stats:
            rr = dict(row)
            rr['model'] = spec.name
            all_layers.append(rr)

        torch.save(
            {
                'spec': spec.__dict__,
                'state_dict': {
                    k: v.detach().cpu()
                    for k, v in model.state_dict().items()
                },
            },
            out / f'{spec.name}_checkpoint.pt',
        )

        row = {
            'model': spec.name,
            'd_model': spec.d_model,
            'n_layer': spec.n_layer,
            'n_head': spec.n_head,
            'd_ff': 4 * spec.d_model,
            'QKV_L': spec.d_model,
            'AttentionOut_L': spec.d_model,
            'FFN1_L': spec.d_model,
            'FFN2_L': 4 * spec.d_model,
            'LMHead_L': spec.d_model,
            'N_total': n_total,
            'N_nonembedding': n_nonembedding,
            'best_val_loss': best,
            'digital_loss': digital_loss,
            'digital_perplexity': digital_ppl,
            'realistic_photonic_loss': photonic_loss,
            'realistic_photonic_perplexity': photonic_ppl,
            'delta_loss_realistic_minus_digital': (
                photonic_loss - digital_loss
            ),
            **nb_metrics,
        }
        rows.append(row)
        write_csv(out / 'scaling_summary.csv', rows)

        print(
            f'digital={digital_loss:.4f} '
            f'realistic={photonic_loss:.4f} '
            f'delta={photonic_loss - digital_loss:+.4f} '
            f'Nb_sys={nb_metrics["Nb_system_rms"]:.3f}'
        )

    ns = [r['N_nonembedding'] for r in rows]
    fits = {
        'digital': fit_power_law(
            ns,
            [r['digital_loss'] for r in rows],
        ),
        'realistic_photonic': fit_power_law(
            ns,
            [r['realistic_photonic_loss'] for r in rows],
        ),
    }

    (out / 'fit_summary.json').write_text(
        json.dumps(fits, indent=2),
        encoding='utf-8',
    )
    write_csv(
        out / 'all_realistic_layer_stats.csv',
        all_layers,
    )
    plot_all(rows, out)

    print('\nSaved to', out)
    print(json.dumps(fits, indent=2))
    return out


def main():
    args = parse_args()
    run_experiment(
        args.config,
        text_path=args.text,
        data_dir=args.data_dir,
    )


if __name__ == '__main__':
    main()
