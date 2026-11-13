# Hybrid Photonic–Electronic Transformer Simulator

Research code for studying the **scaling characteristics of hybrid photonic–electronic decoder-only Transformer language models under hardware nonidealities**.

The repository trains compact Transformer language models digitally and evaluates the same trained checkpoints with a hardware-aware photonic linear-computation model. The main goal is to connect model scale and architecture to photonic numerical error and task-level language-model loss.

## Research workflow

The central analysis follows

\[
N
\rightarrow d_{\mathrm{model}}
\rightarrow L
\rightarrow \text{photonic hardware error}
\rightarrow N_{b,\mathrm{sys}}
\rightarrow \mathcal{L}_{\mathrm{CE}} .
\]

where:

- \(N\): non-embedding parameter count,
- \(d_{\mathrm{model}}\): Transformer hidden width,
- \(L\): optical dot-product length,
- \(N_{b,\mathrm{sys}}\): system effective precision derived from final-logit dynamic range and RMSE,
- \(\mathcal{L}_{\mathrm{CE}}\): test cross-entropy loss.

The study also includes a model-shape analysis of feed-forward ratio, depth/width aspect ratio, and attention-head dimension.

## Hybrid mapping

The following linear operations can be mapped to the photonic engine during inference:

- QKV projection
- attention output projection
- FFN1
- FFN2
- language-model head

The following operations remain electronic:

- token and positional embeddings
- LayerNorm
- \(QK^\mathsf{T}\)
- softmax
- \(A V\)
- residual additions
- GELU

Training is digital. Photonic hardware nonidealities are introduced during inference.

## Dataset and tokenization

The thesis experiments use **WikiText-103 raw** with its official `train`, `validation`, and `test` splits.

The code uses a fixed UTF-8 byte-level tokenizer:

- vocabulary size: 256
- one byte = one token
- reported language-model loss: **nats per byte token**

The byte-level representation is intentional: it avoids tokenizer-training effects and keeps the tokenization rule fixed across all model sizes. Loss values should therefore not be directly compared with results reported using BPE or other subword tokenizers.

## Installation

Python 3.10+ is recommended.

```powershell
python -m pip install -r requirements.txt
```

For AMD ROCm systems, install a PyTorch build appropriate for your ROCm environment separately if required.

## Download WikiText-103

```powershell
python download_wikitext103.py
```

This creates:

```text
data/wikitext103/
├── train.txt
├── validation.txt
└── test.txt
```

The three files preserve the source dataset's official split boundaries.

## Main paper experiment

```powershell
python run_all.py `
  --config "configs\all_in_one_paper.json" `
  --data-dir "data\wikitext103"
```

`run_all.py` performs:

1. digital training for the S01–S10 model family,
2. best-validation-checkpoint selection,
3. digital and realistic-photonic baseline evaluation,
4. hardware sensitivity analysis using the same trained checkpoints,
5. system-effective-precision analysis.

Hardware sweeps do **not** retrain the Transformer checkpoints.

A backward-compatible one-file corpus mode is still available:

```powershell
python run_all.py `
  --config "configs\all_in_one_paper.json" `
  --text "corpus.txt"
```

In `--text` mode only, the legacy 90/5/5 split is applied.

## Main model family

| Model | `d_model` | `n_layer` | `n_head` | `d_ff` |
|---|---:|---:|---:|---:|
| S01 | 16 | 4 | 4 | 64 |
| S02 | 32 | 4 | 4 | 128 |
| S03 | 64 | 4 | 4 | 256 |
| S04 | 96 | 4 | 4 | 384 |
| S05 | 128 | 4 | 4 | 512 |
| S06 | 192 | 4 | 4 | 768 |
| S07 | 256 | 4 | 4 | 1024 |
| S08 | 384 | 4 | 4 | 1536 |
| S09 | 512 | 4 | 4 | 2048 |
| S10 | 768 | 4 | 4 | 3072 |

The exact non-embedding parameter count is computed from each instantiated model. The approximation

\[
N \approx 12\,n_{\mathrm{layer}}d_{\mathrm{model}}^2
\]

is used only as a scaling interpretation.

## Baseline photonic configuration

| Parameter | Baseline |
|---|---:|
| DAC resolution | 8 bit |
| ADC resolution | 12 bit |
| Input calibration percentile | 99.9% |
| Laser power | 3.05 mW |
| Insertion loss | 3 dB |
| Photodetector responsivity | 1.003 A/W |
| Symbol rate | 4 GHz |
| Integration capacitance | 100 pF |
| ADC full-scale voltage | ±0.335 V |
| MZM half-wave voltage | 2.4 V |
| MZM extinction ratio | 20 dB |
| MZM bias errors | 0 rad |
| MZM phase-noise std. | 0.001 rad |
| RIN | -140 dB/Hz |
| Thermal-current-noise density | \(10^{-12}\) A/\(\sqrt{\mathrm{Hz}}\) |
| Temperature | 300 K |
| Noise scale | 1.0 |
| Photonic LM head | enabled |

The simulator includes shot noise, RIN, thermal-current noise, and \(kTC\) noise.

## System effective precision

The main system-level numerical-fidelity metric is calculated from the final Transformer logits.

Let \(y^{(d)}\) be the ideal digital logits and \(y^{(p)}\) the realistic photonic logits:

\[
\mathrm{RMSE}_{\mathrm{sys}}
=
\sqrt{
\frac{1}{M}
\sum_{i=1}^{M}
\left(
y_i^{(p)}-y_i^{(d)}
\right)^2
}.
\]

Then

\[
N_{b,\mathrm{sys}}
=
\log_2
\left(
\frac{
y_{\max}^{(d)}-y_{\min}^{(d)}
}{
\mathrm{RMSE}_{\mathrm{sys}}
}
\right).
\]

This is a reference-based **system metric**. It is not identical to the nominal ADC bit depth or conventional converter ENOB.

## Sensitivity experiments

The paper-oriented configuration includes:

- ADC resolution: 2–16 bit
- DAC resolution: 2–16 bit
- optical-power sweep: 1 µW to 100 mW
- stochastic-noise scaling
- noise-source ablation
- repeated photonic evaluation (`hardware_eval_repeats = 5`)
- controlled \(N_{b,\mathrm{sys}}\)-versus-loss study
- model scaling under fixed \(N_{b,\mathrm{sys}}\)

To rerun sensitivity analysis on an existing training run:

```powershell
python run_sensitivity.py `
  --run "runs\YOUR_RUN" `
  --config "configs\sensitivity_final.json" `
  --data-dir "data\wikitext103"
```

Use the same dataset files that were used to train the checkpoints.

## Model-shape study

```powershell
python run_shape_study.py `
  --config "configs\all_in_one_paper.json" `
  --shape-config "configs\shape_study.json" `
  --data-dir "data\wikitext103"
```

The shape study evaluates:

- \(d_{\mathrm{ff}}/d_{\mathrm{model}}\)
- \(d_{\mathrm{model}}/n_{\mathrm{layer}}\)
- \(d_{\mathrm{head}}=d_{\mathrm{model}}/n_{\mathrm{head}}\)

See [`README_SHAPE_STUDY.md`](README_SHAPE_STUDY.md) for details.

## Outputs

A main run creates a timestamped directory under `runs/`, including the resolved configuration, dataset metadata, checkpoints, CSV summaries, and figures.

Important outputs include:

```text
runs/<timestamp>/
├── all_in_one_config_resolved.json
├── config_resolved.json
├── dataset_info.json
├── scaling_summary.csv
├── loss_vs_N.png
├── delta_loss_vs_N.png
├── Nb_system_vs_N.png
└── sensitivity/
    ├── sensitivity_summary.csv
    ├── adc_vs_loss.csv
    ├── dac_vs_loss.csv
    ├── power_vs_loss.csv
    ├── noise_scale_vs_loss.csv
    ├── noise_source_ablation.csv
    ├── Nb_system_vs_loss.csv
    ├── N_vs_loss_by_Nb_system.csv
    ├── power_vs_loss.png
    ├── power_vs_Nb_system.png
    ├── Nb_system_vs_loss.png
    └── N_vs_loss_by_Nb_system.png
```

Shape-study outputs include:

```text
ffn_ratio.csv
aspect_ratio.csv
head_dimension.csv
shape_study_all.csv
ffn_ratio_vs_loss.png
aspect_ratio_vs_loss.png
head_dimension_vs_loss.png
shape_study_digital_vs_photonic_3panel.png
```

## Reproducibility

Each paper run should preserve:

- the exact resolved JSON configuration,
- `dataset_info.json`,
- random seed,
- software environment,
- final CSV files used to generate thesis figures,
- the Git commit hash used for the run.

The repository intentionally excludes large checkpoints, generated runs, and downloaded WikiText-103 files.

For a thesis freeze, copy the final configuration and figure-source CSVs into `reproduction/` and record the corresponding Git commit hash.

## Tests

```powershell
python -m pytest -q
```

The test suite is intended for numerical sanity checks of the photonic core. Additional regression tests for system effective precision and dataset split handling are recommended as the project evolves.

## Repository structure

```text
.
├── analysis/
├── configs/
│   ├── all_in_one_paper.json
│   ├── sensitivity_final.json
│   └── shape_study.json
├── data/
│   └── text.py
├── models/
│   └── transformer.py
├── photonic/
├── tests/
├── download_wikitext103.py
├── run_all.py
├── run_experiment.py
├── run_sensitivity.py
├── run_shape_study.py
├── train_eval.py
└── requirements.txt
```

## Dataset source

WikiText-103 is obtained through the `Salesforce/wikitext` dataset, configuration `wikitext-103-raw-v1`. The dataset is not redistributed by this repository.

## Thesis

**Scaling Characteristics of Hybrid Photonic–Electronic Transformer Language Models under Hardware Nonidealities**

The code is research software intended to support the numerical experiments and reproducibility of the thesis. It should not be interpreted as a circuit-level or electromagnetic simulation of a fabricated photonic system.

## Citation

If you use this repository, please cite the repository using `CITATION.cff`. A thesis or paper citation can be added after the final manuscript is archived.
