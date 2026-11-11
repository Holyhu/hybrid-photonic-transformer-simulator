#!/usr/bin/env python3
"""ADC / DAC / noise / optical-power sensitivity analysis with system-level effective precision.

Reuses already-trained checkpoints and supports separate model lists for
ADC, DAC, noise, power, and noise-source-ablation sweeps.
"""

import argparse
import csv
import json
import math
import random
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from data.text import ByteTokenizer, dataset_info, load_data_splits, sequential_batches
from models.transformer import TinyGPT, ModelSpec, non_embedding_parameter_count
from photonic.config import PhotonicConfig
from photonic.core import PhotonicLinearEngine
from photonic.stats import PHOTO_STATS
from train_eval import evaluate, calibrate


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--run', required=True, help='Run directory containing checkpoints/config_resolved.json')
    p.add_argument('--config', required=True, help='Sensitivity JSON config')
    p.add_argument(
        '--text',
        default=None,
        help='Backward-compatible single UTF-8 corpus used for training',
    )
    p.add_argument(
        '--data-dir',
        default=None,
        help=(
            'Directory containing official train.txt, validation.txt and '
            'test.txt. Must match the dataset used to train the checkpoints.'
        ),
    )
    p.add_argument('--out-dir', default=None, help='Optional exact sensitivity output directory')
    return p.parse_args()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(name):
    return torch.device('cuda' if name != 'cpu' and torch.cuda.is_available() else 'cpu')


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
    """Evaluate loss/perplexity and return final logits on the same test batches."""
    model.eval()
    losses = []
    logits_all = []

    for x, y in sequential_batches(
        data, batch_size, block_size, device, max_batches
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
    logits = torch.cat(logits_all, dim=0)
    return loss, math.exp(min(loss, 20.0)), logits


def write_csv(path, rows):
    if not rows:
        return
    fieldnames, seen = [], set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with open(path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def mean_std(xs):
    a = np.asarray(xs, dtype=float)
    return float(a.mean()), float(a.std(ddof=1)) if len(a) > 1 else 0.0


def ordered_unique(xs):
    out, seen = [], set()
    for x in xs:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def evaluate_photonic_repeats(
    model,
    test,
    train_cfg,
    device,
    calibration,
    pcfg,
    repeats,
    base_seed,
    digital_logits_cpu,
):
    """Evaluate one hardware condition and compute system effective bits.

    Nb_system_rms is defined from the FINAL Transformer logits:

        RMSE_sys = sqrt(mean((y_photonic - y_digital)^2))

        Nb_system_rms = log2(
            (max(y_digital) - min(y_digital)) / RMSE_sys
        )

    The RMSE used for Nb_system_rms is pooled over all repeats, tokens,
    and vocabulary logits, matching the previous complete-system definition.
    """
    losses = []
    ppls = []
    all_sat = []
    rmse_per_repeat = []
    nb_per_repeat = []

    ideal = digital_logits_cpu.double()
    ideal_max = float(ideal.max().item())
    ideal_min = float(ideal.min().item())
    dynamic_range = ideal_max - ideal_min

    if dynamic_range <= 0.0:
        raise ValueError(
            'Digital-logit dynamic range must be positive to compute Nb_system.'
        )

    total_squared_error = 0.0
    total_error_count = 0
    eps = np.finfo(np.float64).tiny

    for rep in range(repeats):
        seed_all(base_seed + rep)
        PHOTO_STATS.reset()

        engine = PhotonicLinearEngine(pcfg, 'realistic')
        loss, ppl, photonic_logits_cpu = evaluate_with_logits(
            model,
            test,
            train_cfg['batch_size'],
            train_cfg['block_size'],
            device,
            train_cfg['test_batches'],
            engine,
            calibration,
            pcfg.photonic_lm_head,
        )

        losses.append(loss)
        ppls.append(ppl)

        actual = photonic_logits_cpu.double()
        if actual.shape != ideal.shape:
            raise RuntimeError(
                f'Photonic logits shape {tuple(actual.shape)} does not match '
                f'digital logits shape {tuple(ideal.shape)}.'
            )

        error = actual - ideal
        squared_error_sum = float(error.square().sum().item())
        error_count = int(error.numel())

        total_squared_error += squared_error_sum
        total_error_count += error_count

        rmse_rep = math.sqrt(squared_error_sum / max(error_count, 1))
        nb_rep = math.log2(dynamic_range / max(rmse_rep, eps))

        rmse_per_repeat.append(rmse_rep)
        nb_per_repeat.append(nb_rep)

        stats = PHOTO_STATS.rows()
        if stats:
            all_sat.append(
                float(np.mean([
                    float(r['mean_adc_saturation_fraction'])
                    for r in stats
                ]))
            )

    loss_mean, loss_std = mean_std(losses)
    ppl_mean, ppl_std = mean_std(ppls)

    system_rmse = math.sqrt(
        total_squared_error / max(total_error_count, 1)
    )
    nb_system_rms = math.log2(
        dynamic_range / max(system_rmse, eps)
    )

    rmse_repeat_mean, rmse_repeat_std = mean_std(rmse_per_repeat)
    nb_repeat_mean, nb_repeat_std = mean_std(nb_per_repeat)
    sat_mean, sat_std = (
        mean_std(all_sat)
        if all_sat
        else (float('nan'), float('nan'))
    )

    return {
        'photonic_loss_mean': loss_mean,
        'photonic_loss_std': loss_std,
        'photonic_perplexity_mean': ppl_mean,
        'photonic_perplexity_std': ppl_std,
        'system_ideal_max': ideal_max,
        'system_ideal_min': ideal_min,
        'system_dynamic_range': dynamic_range,
        'system_error_rms': system_rmse,
        'system_error_rms_repeat_mean': rmse_repeat_mean,
        'system_error_rms_repeat_std': rmse_repeat_std,
        'Nb_system_rms': nb_system_rms,
        'Nb_system_rms_repeat_mean': nb_repeat_mean,
        'Nb_system_rms_repeat_std': nb_repeat_std,
        'mean_adc_saturation_fraction': sat_mean,
        'mean_adc_saturation_fraction_std': sat_std,
    }


def row_for_sweep(model_name, N, digital_loss, sweep, x_name, x_value, result, repeats):
    return {
        'model': model_name,
        'N_nonembedding': N,
        'sweep': sweep,
        x_name: x_value,
        'digital_loss': digital_loss,
        **result,
        'delta_loss_mean': result['photonic_loss_mean'] - digital_loss,
        'hardware_eval_repeats': repeats,
    }


def plot_numeric(rows, x_key, out_prefix, x_label, out_dir):
    if not rows:
        return
    models = ordered_unique([r['model'] for r in rows])

    for metric, ylabel, suffix in [
        ('photonic_loss_mean', 'Photonic test cross-entropy loss', 'loss'),
        ('delta_loss_mean', 'Photonic loss - digital loss', 'delta_loss'),
    ]:
        fig = plt.figure(figsize=(7.4, 5.2))
        ax = fig.add_subplot(111)
        for model_name in models:
            sub = sorted((r for r in rows if r['model'] == model_name), key=lambda z: float(z[x_key]))
            x = [float(r[x_key]) for r in sub]
            y = [float(r[metric]) for r in sub]
            ax.plot(x, y, 'o-', label=model_name)

            if metric == 'photonic_loss_mean':
                yerr = [float(r['photonic_loss_std']) for r in sub]
                if any(v > 0 for v in yerr):
                    ax.errorbar(x, y, yerr=yerr, fmt='none', capsize=2, alpha=0.45)

        if metric == 'delta_loss_mean':
            ax.axhline(0.0, lw=1)

        ax.set_xlabel(x_label)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / f'{out_prefix}_vs_{suffix}.png', dpi=300)
        plt.close(fig)


def plot_power(rows, out_dir):
    if not rows:
        return
    models = ordered_unique([r['model'] for r in rows])

    for metric, ylabel, filename in [
        ('photonic_loss_mean', 'Photonic test cross-entropy loss', 'power_vs_loss.png'),
        ('delta_loss_mean', 'Photonic loss - digital loss', 'power_vs_delta_loss.png'),
    ]:
        fig = plt.figure(figsize=(8.0, 5.5))
        ax = fig.add_subplot(111)

        for model_name in models:
            sub = sorted((r for r in rows if r['model'] == model_name), key=lambda z: float(z['laser_power_W']))
            x_mw = [float(r['laser_power_W']) * 1e3 for r in sub]
            y = [float(r[metric]) for r in sub]
            ax.plot(x_mw, y, 'o-', label=model_name)

            if metric == 'photonic_loss_mean':
                yerr = [float(r['photonic_loss_std']) for r in sub]
                if any(v > 0 for v in yerr):
                    ax.errorbar(x_mw, y, yerr=yerr, fmt='none', capsize=2, alpha=0.35)

        ax.set_xscale('log')
        ax.set_xlabel('Laser power P (mW)')
        ax.set_ylabel(ylabel)
        if metric == 'delta_loss_mean':
            ax.axhline(0.0, lw=1)
        ax.grid(True, alpha=0.25)
        ax.legend(ncol=2, fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / filename, dpi=300)
        plt.close(fig)


def plot_power_vs_nb_system(rows, out_dir):
    if not rows:
        return

    models = ordered_unique([r['model'] for r in rows])

    fig = plt.figure(figsize=(8.0, 5.5))
    ax = fig.add_subplot(111)

    for model_name in models:
        sub = sorted(
            (r for r in rows if r['model'] == model_name),
            key=lambda z: float(z['laser_power_W']),
        )

        x_mw = [float(r['laser_power_W']) * 1e3 for r in sub]
        y = [float(r['Nb_system_rms']) for r in sub]

        ax.plot(x_mw, y, 'o-', label=model_name)

    ax.set_xscale('log')
    ax.set_xlabel('Laser power P (mW)')
    ax.set_ylabel(r'System effective precision $N_{b,\mathrm{sys}}$ (bits)')
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / 'power_vs_Nb_system.png', dpi=300)
    plt.close(fig)


def plot_nb_system_vs_loss(rows, out_prefix, title, out_dir):
    if not rows:
        return

    valid_rows = [
        r for r in rows
        if math.isfinite(float(r['Nb_system_rms']))
        and math.isfinite(float(r['photonic_loss_mean']))
    ]
    if not valid_rows:
        return

    models = ordered_unique([r['model'] for r in valid_rows])

    fig = plt.figure(figsize=(7.4, 5.2))
    ax = fig.add_subplot(111)

    for model_name in models:
        sub = [r for r in valid_rows if r['model'] == model_name]
        x = [float(r['Nb_system_rms']) for r in sub]
        y = [float(r['photonic_loss_mean']) for r in sub]

        ax.scatter(x, y, s=45, label=model_name)

        yerr = [float(r['photonic_loss_std']) for r in sub]
        if any(v > 0 for v in yerr):
            ax.errorbar(
                x, y, yerr=yerr,
                fmt='none', capsize=2, alpha=0.4
            )

    ax.set_xlabel(r'System effective precision $N_{b,\mathrm{sys}}$ (bits)')
    ax.set_ylabel('Photonic test cross-entropy loss')
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / f'{out_prefix}_Nb_system_vs_loss.png', dpi=300)
    plt.close(fig)


def plot_ablation(rows, out_dir):
    if not rows:
        return
    models = ordered_unique([r['model'] for r in rows])
    sources = ordered_unique([r['noise_source'] for r in rows])

    fig = plt.figure(figsize=(8.2, 5.2))
    ax = fig.add_subplot(111)
    xpos = np.arange(len(sources), dtype=float)

    for model_name in models:
        by = {r['noise_source']: r for r in rows if r['model'] == model_name}
        y = [float(by[source]['delta_loss_mean']) for source in sources]
        ax.plot(xpos, y, 'o-', label=model_name)

    ax.axhline(0.0, lw=1)
    ax.set_xticks(xpos)
    ax.set_xticklabels(sources, rotation=25, ha='right')
    ax.set_ylabel('Photonic loss - digital loss')
    ax.set_xlabel('Enabled random-noise source')
    ax.grid(True, axis='y', alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / 'noise_source_ablation_vs_delta_loss.png', dpi=300)
    plt.close(fig)



@torch.no_grad()
def collect_test_targets(
    data,
    batch_size,
    block_size,
    device,
    max_batches,
):
    """Collect exactly the target tokens used by evaluate_with_logits()."""
    targets = []

    for _, y in sequential_batches(
        data,
        batch_size,
        block_size,
        device,
        max_batches,
    ):
        targets.append(
            y.detach().long().cpu()
        )

    if not targets:
        raise RuntimeError(
            'No target batches were produced for the Nb_system study.'
        )

    return torch.cat(targets, dim=0)


def make_unit_rms_gaussian(reference, seed):
    """Generate zero-mean Gaussian error with empirical RMS exactly equal to 1."""
    generator = torch.Generator(device='cpu')
    generator.manual_seed(int(seed))

    z = torch.randn(
        reference.shape,
        generator=generator,
        dtype=torch.float32,
    )

    z = z - z.mean()

    rms = torch.sqrt(
        torch.mean(z.square())
    )

    if not torch.isfinite(rms) or float(rms) <= 0.0:
        raise RuntimeError(
            'Could not generate a valid Gaussian error pattern.'
        )

    return z / rms


def run_pure_nb_loss_study(
    model_name,
    N_nonembedding,
    digital_logits_cpu,
    targets_cpu,
    digital_loss,
    sens_cfg,
    train_cfg,
):
    """Pure task-level Nb_system study for one fixed model.

    No ADC/DAC/MZM/photonic hardware is used here.
    Zero-mean Gaussian error is injected directly into the final digital
    logits. Only the error RMS changes.

        RMSE_target = dynamic_range / 2**Nb_target
        y_perturbed = y_digital + error
        Nb_system = log2(dynamic_range / RMSE_actual)

    Calling this function for S01--S10 enables model-scaling curves under
    fixed system-effective-precision conditions.
    """
    if not sens_cfg.get('enable_nb_loss_study', True):
        return []

    target_values = [
        float(v)
        for v in sens_cfg.get(
            'nb_loss_target_values',
            [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0],
        )
    ]

    repeats = int(
        sens_cfg.get(
            'nb_loss_repeats',
            sens_cfg.get('hardware_eval_repeats', 5),
        )
    )
    if repeats < 1:
        raise ValueError('nb_loss_repeats must be >= 1')

    # Give each model its own reproducible Gaussian directions.
    base_seed = int(
        sens_cfg.get(
            'nb_loss_seed',
            int(train_cfg['seed']) + 900000,
        )
    ) + int(N_nonembedding % 100000)

    ideal = digital_logits_cpu.float()
    ideal_max = float(ideal.max().item())
    ideal_min = float(ideal.min().item())
    dynamic_range = ideal_max - ideal_min

    if not math.isfinite(dynamic_range) or dynamic_range <= 0.0:
        raise RuntimeError(
            'Digital-logit dynamic range must be positive for Nb_system study.'
        )

    print('\n' + '-' * 76)
    print(
        f'[Pure Nb_system study] {model_name}: '
        'zero-mean Gaussian error on final logits'
    )
    print(
        f'N={N_nonembedding}, digital loss={digital_loss:.6f}, '
        f'dynamic range={dynamic_range:.6f}, repeats={repeats}'
    )

    rows = []

    # Within one model/repeat, use the same Gaussian direction for every
    # target Nb. Therefore only the error amplitude changes.
    unit_errors = [
        make_unit_rms_gaussian(
            ideal,
            base_seed + rep,
        )
        for rep in range(repeats)
    ]

    eps = np.finfo(np.float64).tiny

    for target_nb in target_values:
        target_rmse = dynamic_range / (2.0 ** target_nb)

        losses = []
        achieved_nbs = []
        achieved_rmses = []

        for unit_error in unit_errors:
            error = unit_error * target_rmse
            perturbed = ideal + error

            loss = float(
                F.cross_entropy(
                    perturbed.reshape(
                        -1,
                        perturbed.shape[-1],
                    ),
                    targets_cpu.reshape(-1),
                ).item()
            )

            actual_error = perturbed.double() - ideal.double()

            rmse = float(
                torch.sqrt(
                    torch.mean(actual_error.square())
                ).item()
            )

            achieved_nb = math.log2(
                dynamic_range / max(rmse, eps)
            )

            losses.append(loss)
            achieved_rmses.append(rmse)
            achieved_nbs.append(achieved_nb)

        loss_mean, loss_std = mean_std(losses)
        nb_mean, nb_std = mean_std(achieved_nbs)
        rmse_mean, rmse_std = mean_std(achieved_rmses)

        row = {
            'model': model_name,
            'N_nonembedding': N_nonembedding,
            'error_model': 'zero_mean_gaussian_final_logits',
            'target_Nb_system': target_nb,
            'target_RMSE': target_rmse,
            'achieved_Nb_system_mean': nb_mean,
            'achieved_Nb_system_std': nb_std,
            'achieved_RMSE_mean': rmse_mean,
            'achieved_RMSE_std': rmse_std,
            'system_ideal_max': ideal_max,
            'system_ideal_min': ideal_min,
            'system_dynamic_range': dynamic_range,
            'digital_loss': digital_loss,
            'test_loss_mean': loss_mean,
            'test_loss_std': loss_std,
            'delta_loss_mean': loss_mean - digital_loss,
            'repeats': repeats,
        }
        rows.append(row)

        print(
            f'Nb target={target_nb:>4.1f}  '
            f'Nb achieved={nb_mean:.4f}  '
            f'RMSE={rmse_mean:.6g}  '
            f'loss={loss_mean:.6f}  '
            f'delta={loss_mean - digital_loss:+.6f}'
        )

    return rows


def plot_pure_nb_vs_loss(rows, reference_model, out_dir):
    """Plot Nb_system vs Loss for one representative model, normally S10."""
    sub = [
        r for r in rows
        if r['model'] == reference_model
    ]
    if not sub:
        return

    sub = sorted(
        sub,
        key=lambda r: float(r['achieved_Nb_system_mean'])
    )

    x = [
        float(r['achieved_Nb_system_mean'])
        for r in sub
    ]
    y = [
        float(r['test_loss_mean'])
        for r in sub
    ]
    yerr = [
        float(r['test_loss_std'])
        for r in sub
    ]

    digital_loss = float(sub[0]['digital_loss'])

    fig = plt.figure(figsize=(7.4, 5.2))
    ax = fig.add_subplot(111)

    ax.errorbar(
        x,
        y,
        yerr=yerr,
        fmt='o-',
        capsize=3,
        label=f'{reference_model}: Gaussian output error',
    )

    ax.axhline(
        digital_loss,
        lw=1,
        linestyle='--',
        label='Digital reference',
    )

    ax.set_xlabel(
        r'System effective precision $N_{b,\mathrm{sys}}$ (bits)'
    )
    ax.set_ylabel(
        'Test cross-entropy loss (nats/token)'
    )
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()

    fig.savefig(
        out_dir / 'Nb_system_vs_loss.png',
        dpi=300,
    )
    plt.close(fig)

    fig = plt.figure(figsize=(7.4, 5.2))
    ax = fig.add_subplot(111)

    ax.plot(
        x,
        [
            float(r['delta_loss_mean'])
            for r in sub
        ],
        'o-',
    )

    ax.axhline(0.0, lw=1)
    ax.set_xlabel(
        r'System effective precision $N_{b,\mathrm{sys}}$ (bits)'
    )
    ax.set_ylabel(
        'Loss increase relative to digital reference'
    )
    ax.grid(True, alpha=0.25)
    fig.tight_layout()

    fig.savefig(
        out_dir / 'Nb_system_vs_delta_loss.png',
        dpi=300,
    )
    plt.close(fig)


def plot_scaling_under_nb(rows, sens_cfg, out_dir):
    """Plot N vs loss under several fixed target Nb_system conditions.

    This is the direct analogue of the existing ADC/DAC model-scaling plots,
    except each colored curve represents a controlled system-effective
    precision instead of a converter bit depth.
    """
    if not rows:
        return

    unique_models = sorted({r['model'] for r in rows})
    unique_N = sorted({float(r['N_nonembedding']) for r in rows})
    if len(unique_N) < 2:
        print(
            '\n[WARNING] N_vs_loss_by_Nb_system.png requires at least two model sizes, '
            f'but only {len(unique_models)} model(s) were evaluated: {unique_models}. '
            'Set nb_loss_models to S01--S10 (or another multi-model list). '
            'The one-point scaling figure is not generated.'
        )
        return

    requested_nb = [
        float(v)
        for v in sens_cfg.get(
            'nb_scaling_plot_values',
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0],
        )
    ]

    models = ordered_unique([r['model'] for r in rows])

    # Digital reference: one point per model.
    digital_points = []
    for model_name in models:
        model_rows = [
            r for r in rows
            if r['model'] == model_name
        ]
        if not model_rows:
            continue
        r0 = model_rows[0]
        digital_points.append(
            (
                float(r0['N_nonembedding']),
                float(r0['digital_loss']),
            )
        )

    digital_points.sort(key=lambda t: t[0])

    fig = plt.figure(figsize=(8.6, 5.8))
    ax = fig.add_subplot(111)

    if digital_points:
        ax.plot(
            [p[0] for p in digital_points],
            [p[1] for p in digital_points],
            'o-',
            linewidth=2.6,
            label='Digital',
        )

    for target_nb in requested_nb:
        sub = [
            r for r in rows
            if abs(
                float(r['target_Nb_system'])
                - target_nb
            ) < 1e-9
        ]

        sub = sorted(
            sub,
            key=lambda r: float(r['N_nonembedding'])
        )

        if not sub:
            continue

        x = [
            float(r['N_nonembedding'])
            for r in sub
        ]
        y = [
            float(r['test_loss_mean'])
            for r in sub
        ]
        yerr = [
            float(r['test_loss_std'])
            for r in sub
        ]

        label_value = (
            str(int(target_nb))
            if float(target_nb).is_integer()
            else f'{target_nb:g}'
        )

        ax.plot(
            x,
            y,
            'o-',
            label=rf'$N_{{b,\mathrm{{sys}}}}={label_value}$ bit',
        )

        if any(v > 0 for v in yerr):
            ax.errorbar(
                x,
                y,
                yerr=yerr,
                fmt='none',
                capsize=2,
                alpha=0.35,
            )

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(
        r'Non-embedding parameter count $N$'
    )
    ax.set_ylabel(
        'Test cross-entropy loss (nats/token)'
    )
    ax.set_title(
        r'Model scaling under different system effective precisions'
    )
    ax.grid(True, which='both', alpha=0.25)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()

    fig.savefig(
        out_dir / 'N_vs_loss_by_Nb_system.png',
        dpi=300,
    )
    plt.close(fig)

def _load_config(config):
    if isinstance(config, (str, Path)):
        return json.loads(Path(config).read_text(encoding='utf-8'))
    return dict(config)


def _validate_models(model_names, specs, group_name):
    missing = [m for m in model_names if m not in specs]
    if missing:
        raise ValueError(f'{group_name}: models not found in training run config: {missing}')


def run_sensitivity(run_dir, config, text_path=None, data_dir=None, out_dir=None):
    run_dir = Path(run_dir)
    sens_cfg = _load_config(config)
    train_cfg = json.loads((run_dir / 'config_resolved.json').read_text(encoding='utf-8'))
    device = choose_device(train_cfg.get('device', 'auto'))

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
    base_pcfg = PhotonicConfig(**train_cfg['photonic'])
    specs = {s['name']: ModelSpec(**s) for s in train_cfg['model_specs']}

    # Separate model lists for each experiment.
    default_models = sens_cfg.get('models') or list(specs)
    adc_models = sens_cfg.get('adc_models') or default_models
    dac_models = sens_cfg.get('dac_models') or default_models
    noise_models = sens_cfg.get('noise_models') or default_models
    ablation_models = sens_cfg.get('ablation_models') or default_models

    # Your current JSON has no power_models yet. To make it work immediately,
    # power_models falls back to adc_models (S01-S10 in your JSON).
    power_models = sens_cfg.get('power_models') or sens_cfg.get('adc_models') or default_models

    _validate_models(adc_models, specs, 'adc_models')
    _validate_models(dac_models, specs, 'dac_models')
    _validate_models(noise_models, specs, 'noise_models')
    _validate_models(ablation_models, specs, 'ablation_models')
    _validate_models(power_models, specs, 'power_models')

    nb_loss_models = (
        sens_cfg.get('nb_loss_models')
        or [sens_cfg.get('nb_loss_model', 'S10')]
    )
    nb_loss_reference_model = sens_cfg.get(
        'nb_loss_reference_model',
        'S10',
    )

    if sens_cfg.get('enable_nb_loss_study', True):
        _validate_models(
            nb_loss_models,
            specs,
            'nb_loss_models',
        )
        _validate_models(
            [nb_loss_reference_model],
            specs,
            'nb_loss_reference_model',
        )

    requested = ordered_unique(
        adc_models
        + dac_models
        + noise_models
        + ablation_models
        + power_models
        + (
            nb_loss_models
            if sens_cfg.get('enable_nb_loss_study', True)
            else []
        )
    )

    repeats = int(sens_cfg.get('hardware_eval_repeats', 3))
    if repeats < 1:
        raise ValueError('hardware_eval_repeats must be >= 1')

    # Controlled settings.
    adc_fixed_dac_bits = int(sens_cfg.get('adc_sweep_fixed_dac_bits', 16))
    adc_fixed_noise_scale = float(sens_cfg.get('adc_sweep_fixed_noise_scale', 1.0))

    dac_fixed_adc_bits = int(sens_cfg.get('dac_sweep_fixed_adc_bits', 16))
    dac_fixed_noise_scale = float(sens_cfg.get('dac_sweep_fixed_noise_scale', 1.0))

    power_fixed_adc_bits = int(sens_cfg.get('power_sweep_fixed_adc_bits', 16))
    power_fixed_dac_bits = int(sens_cfg.get('power_sweep_fixed_dac_bits', 16))
    power_fixed_noise_scale = float(sens_cfg.get('power_sweep_fixed_noise_scale', 1.0))

    noise_fixed_adc_bits = sens_cfg.get('noise_sweep_fixed_adc_bits', None)
    noise_fixed_dac_bits = sens_cfg.get('noise_sweep_fixed_dac_bits', None)

    if out_dir is None:
        out = run_dir / ('sensitivity_' + datetime.now().strftime('%Y%m%d_%H%M%S'))
    else:
        out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    resolved_config = dict(sens_cfg)
    resolved_config.update({
        'resolved_adc_models': adc_models,
        'resolved_dac_models': dac_models,
        'resolved_noise_models': noise_models,
        'resolved_ablation_models': ablation_models,
        'resolved_power_models': power_models,
        'resolved_nb_loss_models': nb_loss_models,
        'resolved_nb_loss_reference_model': nb_loss_reference_model,
        'resolved_adc_sweep_fixed_dac_bits': adc_fixed_dac_bits,
        'resolved_adc_sweep_fixed_noise_scale': adc_fixed_noise_scale,
        'resolved_dac_sweep_fixed_adc_bits': dac_fixed_adc_bits,
        'resolved_dac_sweep_fixed_noise_scale': dac_fixed_noise_scale,
        'resolved_power_sweep_fixed_adc_bits': power_fixed_adc_bits,
        'resolved_power_sweep_fixed_dac_bits': power_fixed_dac_bits,
        'resolved_power_sweep_fixed_noise_scale': power_fixed_noise_scale,
    })

    (out / 'sensitivity_config_resolved.json').write_text(json.dumps(resolved_config, indent=2), encoding='utf-8')
    (out / 'base_photonic_config.json').write_text(json.dumps(asdict(base_pcfg), indent=2), encoding='utf-8')
    (out / 'dataset_info.json').write_text(
        json.dumps(ds_info, indent=2),
        encoding='utf-8',
    )

    print('Dataset mode:', ds_info['mode'])
    print('Training dataset size D:', ds_info['train_tokens_D'], 'byte tokens')
    print('Validation tokens:', ds_info['validation_tokens'])
    print('Test tokens:', ds_info['test_tokens'])

    adc_rows, dac_rows, noise_rows, power_rows, ablation_rows, nb_loss_rows, all_rows = [], [], [], [], [], [], []

    for mi, model_name in enumerate(requested):
        spec = specs[model_name]
        ckpt_path = run_dir / f'{model_name}_checkpoint.pt'
        if not ckpt_path.exists():
            raise FileNotFoundError(f'Missing checkpoint: {ckpt_path}')

        print('\n' + '=' * 76)
        print(f'Sensitivity: loading {model_name}: {spec}')

        model = TinyGPT(ByteTokenizer.vocab_size, train_cfg['block_size'], spec).to(device)
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt['state_dict'])
        model.eval()

        digital_loss, digital_ppl, digital_logits_cpu = evaluate_with_logits(
            model,
            test,
            train_cfg['batch_size'],
            train_cfg['block_size'],
            device,
            train_cfg['test_batches'],
        )
        calibration = calibrate(model, val, train_cfg, device, base_pcfg.input_percentile)
        N = non_embedding_parameter_count(model)
        eval_seed = int(train_cfg['seed']) + 500000 + 10000 * mi

        print(f'Digital reference loss={digital_loss:.6f}, ppl={digital_ppl:.3f}')

        # Pure Nb_system study:
        # run the same controlled Gaussian-error experiment for S01--S10.
        # This generates both:
        #   1) Nb_system vs Loss for a representative model (default S10)
        #   2) N vs Loss under different fixed Nb_system conditions.
        if (
            sens_cfg.get('enable_nb_loss_study', True)
            and model_name in nb_loss_models
        ):
            targets_cpu = collect_test_targets(
                test,
                train_cfg['batch_size'],
                train_cfg['block_size'],
                device,
                train_cfg['test_batches'],
            )

            model_nb_rows = run_pure_nb_loss_study(
                model_name=model_name,
                N_nonembedding=N,
                digital_logits_cpu=digital_logits_cpu,
                targets_cpu=targets_cpu,
                digital_loss=digital_loss,
                sens_cfg=sens_cfg,
                train_cfg=train_cfg,
            )

            nb_loss_rows.extend(model_nb_rows)


        # ADC sweep: DAC=16, noise_scale=1 by default.
        if model_name in adc_models:
            print(f'\n[ADC sweep] {model_name}: DAC={adc_fixed_dac_bits} bit, noise_scale={adc_fixed_noise_scale:g}')
            for bits in sens_cfg.get('adc_bits', []):
                bits = int(bits)
                pcfg = replace(
                    base_pcfg,
                    adc_bits=bits,
                    dac_bits=adc_fixed_dac_bits,
                    noise_scale=adc_fixed_noise_scale,
                )
                res = evaluate_photonic_repeats(model, test, train_cfg, device, calibration, pcfg, repeats, eval_seed, digital_logits_cpu)
                row = row_for_sweep(model_name, N, digital_loss, 'adc_bits', 'adc_bits', bits, res, repeats)
                row['fixed_dac_bits'] = adc_fixed_dac_bits
                row['fixed_noise_scale'] = adc_fixed_noise_scale
                adc_rows.append(row)
                all_rows.append(row)
                print(
                    f'ADC={bits:>2} bit  DAC={adc_fixed_dac_bits:>2}  noise={adc_fixed_noise_scale:g}  '
                    f'loss={res["photonic_loss_mean"]:.6f}  delta={row["delta_loss_mean"]:+.6f}'
                )

        # DAC sweep: ADC=16, noise_scale=1 by default.
        if model_name in dac_models:
            print(f'\n[DAC sweep] {model_name}: ADC={dac_fixed_adc_bits} bit, noise_scale={dac_fixed_noise_scale:g}')
            for bits in sens_cfg.get('dac_bits', []):
                bits = int(bits)
                pcfg = replace(
                    base_pcfg,
                    dac_bits=bits,
                    adc_bits=dac_fixed_adc_bits,
                    noise_scale=dac_fixed_noise_scale,
                )
                res = evaluate_photonic_repeats(model, test, train_cfg, device, calibration, pcfg, repeats, eval_seed, digital_logits_cpu)
                row = row_for_sweep(model_name, N, digital_loss, 'dac_bits', 'dac_bits', bits, res, repeats)
                row['fixed_adc_bits'] = dac_fixed_adc_bits
                row['fixed_noise_scale'] = dac_fixed_noise_scale
                dac_rows.append(row)
                all_rows.append(row)
                print(
                    f'DAC={bits:>2} bit  ADC={dac_fixed_adc_bits:>2}  noise={dac_fixed_noise_scale:g}  '
                    f'loss={res["photonic_loss_mean"]:.6f}  delta={row["delta_loss_mean"]:+.6f}'
                )

        # Power sweep: ADC=16, DAC=16, noise_scale=1 by default.
        if model_name in power_models:
            power_values = sens_cfg.get('power_values_W', [])
            if not power_values:
                print(
                    f'\n[WARNING] Power sweep requested for {model_name}, but '
                    'power_values_W is missing or empty. Power sweep is skipped.'
                )
            else:
                print(
                    f'\n[Power sweep] {model_name}: ADC={power_fixed_adc_bits} bit, '
                    f'DAC={power_fixed_dac_bits} bit, noise_scale={power_fixed_noise_scale:g}'
                )

            for power_W in power_values:
                power_W = float(power_W)
                if power_W <= 0:
                    raise ValueError('power_values_W must contain positive values because the power plot uses a log axis.')

                pcfg = replace(
                    base_pcfg,
                    laser_power_W=power_W,
                    adc_bits=power_fixed_adc_bits,
                    dac_bits=power_fixed_dac_bits,
                    noise_scale=power_fixed_noise_scale,
                )
                res = evaluate_photonic_repeats(model, test, train_cfg, device, calibration, pcfg, repeats, eval_seed, digital_logits_cpu)
                row = row_for_sweep(model_name, N, digital_loss, 'laser_power', 'laser_power_W', power_W, res, repeats)
                row['laser_power_mW'] = power_W * 1e3
                row['fixed_adc_bits'] = power_fixed_adc_bits
                row['fixed_dac_bits'] = power_fixed_dac_bits
                row['fixed_noise_scale'] = power_fixed_noise_scale
                power_rows.append(row)
                all_rows.append(row)

                print(
                    f'P={power_W * 1e3:>7.3f} mW  ADC={power_fixed_adc_bits:>2}  '
                    f'DAC={power_fixed_dac_bits:>2}  noise={power_fixed_noise_scale:g}  '
                    f'loss={res["photonic_loss_mean"]:.6f}  delta={row["delta_loss_mean"]:+.6f}  '
                    f'Nb_sys={res["Nb_system_rms"]:.3f}  RMSE={res["system_error_rms"]:.6g}  sat={res["mean_adc_saturation_fraction"]:.6f}'
                )

        # Noise-scale sweep.
        if model_name in noise_models:
            for scale in sens_cfg.get('noise_scale', []):
                scale = float(scale)
                updates = {'noise_scale': scale}
                if noise_fixed_adc_bits is not None:
                    updates['adc_bits'] = int(noise_fixed_adc_bits)
                if noise_fixed_dac_bits is not None:
                    updates['dac_bits'] = int(noise_fixed_dac_bits)

                pcfg = replace(base_pcfg, **updates)
                res = evaluate_photonic_repeats(model, test, train_cfg, device, calibration, pcfg, repeats, eval_seed, digital_logits_cpu)
                row = row_for_sweep(model_name, N, digital_loss, 'noise_scale', 'noise_scale', scale, res, repeats)
                if noise_fixed_adc_bits is not None:
                    row['fixed_adc_bits'] = int(noise_fixed_adc_bits)
                if noise_fixed_dac_bits is not None:
                    row['fixed_dac_bits'] = int(noise_fixed_dac_bits)
                noise_rows.append(row)
                all_rows.append(row)
                print(f'Noise scale={scale:>5g}  loss={res["photonic_loss_mean"]:.6f}  delta={row["delta_loss_mean"]:+.6f}')

        # Noise-source ablation.
        if model_name in ablation_models and sens_cfg.get('noise_source_ablation', True):
            variants = {
                'none': replace(base_pcfg, enable_shot_noise=False, enable_rin=False, enable_thermal_noise=False, enable_ktc=False, mzm_phase_noise_std_rad=0.0, noise_scale=1.0),
                'shot': replace(base_pcfg, enable_shot_noise=True, enable_rin=False, enable_thermal_noise=False, enable_ktc=False, mzm_phase_noise_std_rad=0.0, noise_scale=1.0),
                'rin': replace(base_pcfg, enable_shot_noise=False, enable_rin=True, enable_thermal_noise=False, enable_ktc=False, mzm_phase_noise_std_rad=0.0, noise_scale=1.0),
                'thermal': replace(base_pcfg, enable_shot_noise=False, enable_rin=False, enable_thermal_noise=True, enable_ktc=False, mzm_phase_noise_std_rad=0.0, noise_scale=1.0),
                'kTC': replace(base_pcfg, enable_shot_noise=False, enable_rin=False, enable_thermal_noise=False, enable_ktc=True, mzm_phase_noise_std_rad=0.0, noise_scale=1.0),
                'MZM phase': replace(base_pcfg, enable_shot_noise=False, enable_rin=False, enable_thermal_noise=False, enable_ktc=False, mzm_phase_noise_std_rad=base_pcfg.mzm_phase_noise_std_rad, noise_scale=1.0),
                'all': replace(base_pcfg, noise_scale=1.0),
            }

            for source, pcfg in variants.items():
                res = evaluate_photonic_repeats(model, test, train_cfg, device, calibration, pcfg, repeats, eval_seed, digital_logits_cpu)
                row = {
                    'model': model_name,
                    'N_nonembedding': N,
                    'sweep': 'noise_source_ablation',
                    'noise_source': source,
                    'digital_loss': digital_loss,
                    **res,
                    'delta_loss_mean': res['photonic_loss_mean'] - digital_loss,
                    'hardware_eval_repeats': repeats,
                }
                ablation_rows.append(row)
                all_rows.append(row)

    # Save CSVs.
    write_csv(out / 'adc_vs_loss.csv', adc_rows)
    write_csv(out / 'dac_vs_loss.csv', dac_rows)
    write_csv(out / 'noise_scale_vs_loss.csv', noise_rows)
    write_csv(out / 'power_vs_loss.csv', power_rows)
    write_csv(out / 'noise_source_ablation.csv', ablation_rows)
    write_csv(out / 'Nb_system_vs_loss.csv', nb_loss_rows)
    write_csv(out / 'N_vs_loss_by_Nb_system.csv', nb_loss_rows)
    write_csv(out / 'sensitivity_summary.csv', all_rows)

    # Main plots.
    plot_numeric(adc_rows, 'adc_bits', 'adc_bits', 'ADC resolution (bits)', out)
    plot_numeric(dac_rows, 'dac_bits', 'dac_bits', 'DAC resolution (bits)', out)
    plot_numeric(noise_rows, 'noise_scale', 'noise_scale', 'Random-noise standard-deviation multiplier', out)
    plot_power(power_rows, out)
    plot_power_vs_nb_system(power_rows, out)
    plot_ablation(ablation_rows, out)

    # Pure system-effective-precision analysis.
    plot_pure_nb_vs_loss(
        nb_loss_rows,
        nb_loss_reference_model,
        out,
    )
    plot_scaling_under_nb(
        nb_loss_rows,
        sens_cfg,
        out,
    )

    # System-effective-precision diagnostics.
    print('Important: noise_scale multiplies RANDOM-noise standard deviation, not variance.')
    print(
        'For the power sweep, noise_scale=1.0 does not freeze absolute physical noise amplitude; '
        'power-dependent shot/RIN terms may still vary with optical power.'
    )
    if nb_loss_rows:
        print(
            'Pure Nb_system study models:',
            ', '.join(nb_loss_models),
        )
        print(
            'Nb_system-vs-loss reference model:',
            nb_loss_reference_model,
        )
        print(
            'Main Nb figure:',
            out / 'Nb_system_vs_loss.png',
        )
        print(
            'Scaling-under-Nb figure:',
            out / 'N_vs_loss_by_Nb_system.png',
        )
    return out


def main():
    args = parse_args()
    run_sensitivity(
        args.run,
        args.config,
        text_path=args.text,
        data_dir=args.data_dir,
        out_dir=args.out_dir,
    )


if __name__ == '__main__':
    main()
