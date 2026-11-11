#!/usr/bin/env python3
"""Transformer model-shape study: Digital vs Realistic Photonic.

Produces three primary figures:
  1. ffn_ratio_vs_loss.png
  2. aspect_ratio_vs_loss.png
  3. head_dimension_vs_loss.png

Each point trains a digital Transformer, restores the best validation
checkpoint, then evaluates the SAME trained weights in:
  - ideal digital inference
  - realistic photonic inference

The y-axis is absolute test cross-entropy loss (nats/token).
CSV files also include the photonic hardware penalty:
    Delta L = L_photonic - L_digital
"""

import argparse
import csv
import json
import math
import random
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from data.text import (
    ByteTokenizer,
    dataset_info,
    load_data_splits,
    sequential_batches,
)
from models.transformer import (
    TinyGPT,
    ModelSpec,
    total_parameter_count,
    non_embedding_parameter_count,
)
from photonic.config import PhotonicConfig
from photonic.core import PhotonicLinearEngine
from photonic.stats import PHOTO_STATS
from train_eval import train_digital, calibrate


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        '--config',
        required=True,
        help=(
            'Your normal training config / all-in-one config. '
            'Training and photonic settings are reused.'
        ),
    )
    p.add_argument(
        '--shape-config',
        default='configs/shape_study.json',
        help='Shape-study architecture grid.',
    )
    p.add_argument(
        '--data-dir',
        default=None,
        help=(
            'Official split directory containing train.txt, '
            'validation.txt and test.txt.'
        ),
    )
    p.add_argument(
        '--text',
        default=None,
        help='Backward-compatible one-file corpus mode.',
    )
    p.add_argument(
        '--out-dir',
        default=None,
    )
    return p.parse_args()


def load_json(path):
    return json.loads(
        Path(path).read_text(encoding='utf-8')
    )


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


def write_csv(path, rows):
    if not rows:
        return

    fields = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)

    with open(
        path,
        'w',
        newline='',
        encoding='utf-8-sig',
    ) as f:
        w = csv.DictWriter(
            f,
            fieldnames=fields,
        )
        w.writeheader()
        w.writerows(rows)


@torch.no_grad()
def evaluate_with_logits(
    model,
    data,
    cfg,
    device,
    engine=None,
    calibration=None,
    photonic_lm_head=True,
):
    model.eval()
    losses = []
    logits_all = []

    for x, y in sequential_batches(
        data,
        cfg['batch_size'],
        cfg['block_size'],
        device,
        cfg['test_batches'],
    ):
        logits, loss = model(
            x,
            y,
            engine=engine,
            calibration=calibration,
            photonic_lm_head=photonic_lm_head,
        )
        losses.append(
            float(loss.detach().cpu())
        )
        logits_all.append(
            logits.detach().float().cpu()
        )

    if not losses:
        raise RuntimeError(
            'No evaluation batches were produced.'
        )

    loss = float(np.mean(losses))
    logits = torch.cat(
        logits_all,
        dim=0,
    )
    return loss, logits


def calculate_nb_system(
    photonic_logits,
    digital_logits,
):
    ideal = digital_logits.double()
    actual = photonic_logits.double()

    err = actual - ideal
    rmse = float(
        torch.sqrt(
            torch.mean(err.square())
        ).item()
    )

    ideal_max = float(
        ideal.max().item()
    )
    ideal_min = float(
        ideal.min().item()
    )
    dynamic_range = (
        ideal_max - ideal_min
    )

    eps = np.finfo(np.float64).tiny
    nb = math.log2(
        max(dynamic_range, eps)
        / max(rmse, eps)
    )

    return rmse, dynamic_range, nb


def mean_std(values):
    a = np.asarray(values, dtype=float)
    return (
        float(a.mean()),
        float(a.std(ddof=1))
        if len(a) > 1
        else 0.0,
    )


def architecture_key(spec):
    return (
        int(spec.d_model),
        int(spec.n_layer),
        int(spec.n_head),
        int(spec.effective_d_ff),
    )


def metric_for_group(group_name, spec):
    if group_name == 'ffn_ratio':
        return (
            spec.effective_d_ff
            / spec.d_model
        )

    if group_name == 'aspect_ratio':
        return (
            spec.d_model
            / spec.n_layer
        )

    if group_name == 'head_dimension':
        return (
            spec.d_model
            / spec.n_head
        )

    raise ValueError(
        f'Unknown group: {group_name}'
    )


def plot_group(
    rows,
    group_name,
    out,
):
    if not rows:
        return

    labels = {
        'ffn_ratio': (
            r'Feed-forward ratio '
            r'$d_{\mathrm{ff}}/d_{\mathrm{model}}$',
            'ffn_ratio_vs_loss.png',
        ),
        'aspect_ratio': (
            r'Aspect ratio '
            r'$d_{\mathrm{model}}/n_{\mathrm{layer}}$',
            'aspect_ratio_vs_loss.png',
        ),
        'head_dimension': (
            r'Attention head dimension '
            r'$d_{\mathrm{head}}$',
            'head_dimension_vs_loss.png',
        ),
    }

    xlabel, filename = labels[group_name]

    sub = sorted(
        rows,
        key=lambda r: float(r['shape_x']),
    )

    x = [
        float(r['shape_x'])
        for r in sub
    ]
    yd = [
        float(r['digital_loss'])
        for r in sub
    ]
    yp = [
        float(r['photonic_loss_mean'])
        for r in sub
    ]
    yp_err = [
        float(r['photonic_loss_std'])
        for r in sub
    ]

    fig = plt.figure(
        figsize=(7.4, 5.2)
    )
    ax = fig.add_subplot(111)

    ax.plot(
        x,
        yd,
        'o-',
        linewidth=2.0,
        label='Digital',
    )

    ax.errorbar(
        x,
        yp,
        yerr=yp_err,
        fmt='^-',
        linewidth=2.0,
        capsize=3,
        label='Realistic photonic',
    )

    # The grids are approximately geometric, so logarithmic x spacing
    # makes the architecture sweep easier to read.
    ax.set_xscale(
        'log',
        base=2,
    )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(
        'Test cross-entropy loss (nats/token)'
    )
    ax.grid(
        True,
        which='both',
        alpha=0.25,
    )
    ax.legend()
    fig.tight_layout()

    fig.savefig(
        out / filename,
        dpi=300,
    )
    plt.close(fig)


def plot_combined(
    all_rows,
    out,
):
    groups = [
        'ffn_ratio',
        'aspect_ratio',
        'head_dimension',
    ]

    titles = [
        'Feed-forward ratio',
        'Aspect ratio',
        'Attention head dimension',
    ]

    xlabels = [
        r'$d_{\mathrm{ff}}/d_{\mathrm{model}}$',
        r'$d_{\mathrm{model}}/n_{\mathrm{layer}}$',
        r'$d_{\mathrm{head}}$',
    ]

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15.5, 4.6),
    )

    for ax, group, title, xlabel in zip(
        axes,
        groups,
        titles,
        xlabels,
    ):
        sub = sorted(
            [
                r for r in all_rows
                if r['study'] == group
            ],
            key=lambda r: float(r['shape_x']),
        )

        x = [
            float(r['shape_x'])
            for r in sub
        ]
        yd = [
            float(r['digital_loss'])
            for r in sub
        ]
        yp = [
            float(r['photonic_loss_mean'])
            for r in sub
        ]
        yp_err = [
            float(r['photonic_loss_std'])
            for r in sub
        ]

        ax.plot(
            x,
            yd,
            'o-',
            linewidth=1.8,
            label='Digital',
        )
        ax.errorbar(
            x,
            yp,
            yerr=yp_err,
            fmt='^-',
            linewidth=1.8,
            capsize=2,
            label='Realistic photonic',
        )

        ax.set_xscale(
            'log',
            base=2,
        )
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.grid(
            True,
            which='both',
            alpha=0.25,
        )

    axes[0].set_ylabel(
        'Test cross-entropy loss (nats/token)'
    )
    axes[0].legend(
        fontsize=8,
    )

    fig.tight_layout()
    fig.savefig(
        out / 'shape_study_digital_vs_photonic_3panel.png',
        dpi=300,
    )
    plt.close(fig)


def train_and_measure(
    spec,
    cfg,
    train,
    val,
    test,
    pcfg,
    device,
    seed,
    repeats,
    checkpoint_dir,
):
    seed_all(seed)

    print('\n' + '=' * 78)
    print(
        f'Training shape model: {spec.name} | '
        f'd_model={spec.d_model}, '
        f'n_layer={spec.n_layer}, '
        f'n_head={spec.n_head}, '
        f'd_ff={spec.effective_d_ff}'
    )

    model = TinyGPT(
        ByteTokenizer.vocab_size,
        cfg['block_size'],
        spec,
    ).to(device)

    best_val_loss = train_digital(
        model,
        train,
        val,
        cfg,
        device,
    )

    n_total = total_parameter_count(
        model
    )
    n_nonembedding = (
        non_embedding_parameter_count(
            model
        )
    )

    digital_loss, digital_logits = (
        evaluate_with_logits(
            model,
            test,
            cfg,
            device,
        )
    )

    calibration = calibrate(
        model,
        val,
        cfg,
        device,
        pcfg.input_percentile,
    )

    photonic_losses = []
    nb_values = []
    rmse_values = []
    dynamic_ranges = []

    for rep in range(repeats):
        seed_all(
            seed + 100000 + rep
        )
        PHOTO_STATS.reset()

        engine = PhotonicLinearEngine(
            pcfg,
            'realistic',
        )

        p_loss, p_logits = (
            evaluate_with_logits(
                model,
                test,
                cfg,
                device,
                engine=engine,
                calibration=calibration,
                photonic_lm_head=(
                    pcfg.photonic_lm_head
                ),
            )
        )

        rmse, dynamic_range, nb = (
            calculate_nb_system(
                p_logits,
                digital_logits,
            )
        )

        photonic_losses.append(
            p_loss
        )
        nb_values.append(nb)
        rmse_values.append(rmse)
        dynamic_ranges.append(
            dynamic_range
        )

    p_mean, p_std = mean_std(
        photonic_losses
    )
    nb_mean, nb_std = mean_std(
        nb_values
    )
    rmse_mean, rmse_std = mean_std(
        rmse_values
    )

    ckpt_path = (
        checkpoint_dir
        / f'{spec.name}_checkpoint.pt'
    )

    torch.save(
        {
            'spec': asdict(spec),
            'state_dict': {
                k: v.detach().cpu()
                for k, v
                in model.state_dict().items()
            },
        },
        ckpt_path,
    )

    result = {
        'architecture_name': spec.name,
        'd_model': spec.d_model,
        'n_layer': spec.n_layer,
        'n_head': spec.n_head,
        'd_head': (
            spec.d_model
            / spec.n_head
        ),
        'd_ff': spec.effective_d_ff,
        'ffn_ratio': (
            spec.effective_d_ff
            / spec.d_model
        ),
        'aspect_ratio': (
            spec.d_model
            / spec.n_layer
        ),
        'N_total': n_total,
        'N_nonembedding': (
            n_nonembedding
        ),
        'best_val_loss': (
            best_val_loss
        ),
        'digital_loss': (
            digital_loss
        ),
        'photonic_loss_mean': (
            p_mean
        ),
        'photonic_loss_std': (
            p_std
        ),
        'delta_loss': (
            p_mean - digital_loss
        ),
        'photonic_penalty_percent': (
            100.0
            * (p_mean - digital_loss)
            / digital_loss
        ),
        'Nb_system_mean': (
            nb_mean
        ),
        'Nb_system_std': (
            nb_std
        ),
        'system_error_rms_mean': (
            rmse_mean
        ),
        'system_error_rms_std': (
            rmse_std
        ),
        'hardware_eval_repeats': (
            repeats
        ),
    }

    print(
        f'digital={digital_loss:.4f} '
        f'photonic={p_mean:.4f} '
        f'delta={p_mean - digital_loss:+.4f} '
        f'Nb_sys={nb_mean:.3f}'
    )

    return result


def main():
    args = parse_args()

    cfg = load_json(
        args.config
    )
    shape_cfg = load_json(
        args.shape_config
    )

    # all_in_one configs may contain a sensitivity section.
    # Shape training simply ignores it.
    cfg = {
        k: v
        for k, v in cfg.items()
        if k != 'sensitivity'
    }

    device = choose_device(
        cfg.get('device', 'auto')
    )

    train, val, test = (
        load_data_splits(
            text_path=args.text,
            data_dir=args.data_dir,
        )
    )

    if args.out_dir is None:
        out = (
            Path(
                cfg.get(
                    'output_dir',
                    'runs',
                )
            )
            / (
                'shape_study_'
                + datetime.now().strftime(
                    '%Y%m%d_%H%M%S'
                )
            )
        )
    else:
        out = Path(
            args.out_dir
        )

    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint_dir = (
        out / 'checkpoints'
    )
    checkpoint_dir.mkdir(
        exist_ok=True,
    )

    ds_info = dataset_info(
        train,
        val,
        test,
        text_path=args.text,
        data_dir=args.data_dir,
    )

    (
        out / 'dataset_info.json'
    ).write_text(
        json.dumps(
            ds_info,
            indent=2,
        ),
        encoding='utf-8',
    )

    (
        out / 'shape_config_resolved.json'
    ).write_text(
        json.dumps(
            shape_cfg,
            indent=2,
        ),
        encoding='utf-8',
    )

    (
        out / 'training_config_resolved.json'
    ).write_text(
        json.dumps(
            cfg,
            indent=2,
        ),
        encoding='utf-8',
    )

    pcfg = PhotonicConfig(
        **cfg['photonic']
    )

    repeats = int(
        shape_cfg.get(
            'hardware_eval_repeats',
            5,
        )
    )

    groups = [
        'ffn_ratio',
        'aspect_ratio',
        'head_dimension',
    ]

    # Same architecture may appear in multiple panels.
    # Cache it so it is trained only once.
    cache = {}
    all_rows = []

    arch_counter = 0

    for group in groups:
        group_rows = []

        for raw in shape_cfg[group]:
            spec = ModelSpec(**raw)
            key = architecture_key(
                spec
            )

            if key not in cache:
                seed = (
                    int(cfg['seed'])
                    + 1000 * arch_counter
                )
                arch_counter += 1

                cache[key] = (
                    train_and_measure(
                        spec=spec,
                        cfg=cfg,
                        train=train,
                        val=val,
                        test=test,
                        pcfg=pcfg,
                        device=device,
                        seed=seed,
                        repeats=repeats,
                        checkpoint_dir=checkpoint_dir,
                    )
                )
            else:
                print(
                    '\nReusing already-trained '
                    f'architecture for {spec.name}: '
                    f'{key}'
                )

            row = dict(
                cache[key]
            )
            row['study'] = group
            row['study_model_name'] = (
                spec.name
            )
            row['shape_x'] = (
                metric_for_group(
                    group,
                    spec,
                )
            )

            group_rows.append(row)
            all_rows.append(row)

        write_csv(
            out / f'{group}.csv',
            group_rows,
        )

        plot_group(
            group_rows,
            group,
            out,
        )

    write_csv(
        out / 'shape_study_all.csv',
        all_rows,
    )

    plot_combined(
        all_rows,
        out,
    )

    print('\n' + '=' * 78)
    print('SHAPE STUDY COMPLETE')
    print('=' * 78)
    print('Saved to:', out)
    print(
        '1.',
        out / 'ffn_ratio_vs_loss.png',
    )
    print(
        '2.',
        out / 'aspect_ratio_vs_loss.png',
    )
    print(
        '3.',
        out / 'head_dimension_vs_loss.png',
    )
    print(
        'Combined:',
        out / 'shape_study_digital_vs_photonic_3panel.png',
    )


if __name__ == '__main__':
    main()
