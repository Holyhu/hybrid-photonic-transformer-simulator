# Photonic Transformer Scaling + Hardware Sensitivity

This project compares the **same decoder-only Transformer** under:

1. **Digital computation**
2. **Realistic hybrid photonic-electronic computation**

The publication experiment does **not** save an Ideal Photonic curve. Ideal mode remains only inside the unit test as a mathematical sanity check.

The main research chain is:

`model size N -> d_model -> photonic dot-product length L -> photonic error / ENOB -> language-model loss`

Photonic layers: QKV, attention output projection, FFN1, FFN2, and optionally LM head. Softmax, QK^T, A@V, GELU, LayerNorm, residual connections, embeddings and optimizer remain electronic.

For FFN expansion ratio 4:

- QKV / attention output / FFN1 / LM head: `L = d_model`
- FFN2: `L = 4*d_model`

## Recommended: one command for the whole experiment

The new default workflow is **one task**:

`train each model once -> digital/realistic-photonic baseline -> ADC sweep -> DAC sweep -> noise sweep -> noise-source ablation`

All hardware sweeps reuse the exact same trained checkpoints. They do **not** retrain the Transformer for every ADC/DAC/noise point.

### Quick end-to-end check

```powershell
python run_all.py --config configs/all_in_one_demo.json
```

This uses the built-in demo corpus.

### Fine-N quick pipeline check

```powershell
python run_all.py --config configs/all_in_one_fine_quick.json --text corpus.txt
```

### Paper-oriented run

```powershell
python run_all.py --config configs/all_in_one_paper.json --text corpus.txt
```

Everything from one run is saved under one timestamped directory:

```text
runs/
└── 20261014_020000/
    ├── all_in_one_config_resolved.json
    ├── config_resolved.json
    ├── environment.json
    ├── S01_checkpoint.pt
    ├── S02_checkpoint.pt
    ├── ...
    ├── scaling_summary.csv
    ├── all_realistic_layer_stats.csv
    ├── loss_vs_N.png
    ├── delta_loss_vs_N.png
    ├── enob_vs_L.png
    ├── fit_summary.json
    ├── run_all_complete.json
    └── sensitivity/
        ├── adc_vs_loss.csv
        ├── dac_vs_loss.csv
        ├── noise_scale_vs_loss.csv
        ├── noise_source_ablation.csv
        ├── sensitivity_summary.csv
        ├── adc_bits_vs_loss.png
        ├── adc_bits_vs_delta_loss.png
        ├── dac_bits_vs_loss.png
        ├── dac_bits_vs_delta_loss.png
        ├── noise_scale_vs_loss.png
        ├── noise_scale_vs_delta_loss.png
        └── noise_source_ablation_vs_delta_loss.png
```

## One-file configuration

The `all_in_one_*.json` files contain both model/training parameters and the sensitivity study. The sensitivity settings are at the bottom:

```json
"sensitivity": {
  "models": ["S03", "S05", "S07"],
  "hardware_eval_repeats": 5,
  "adc_bits": [4, 5, 6, 7, 8, 10, 12],
  "dac_bits": [4, 5, 6, 7, 8, 10, 12],
  "noise_scale": [0.0, 0.25, 0.5, 1.0, 2.0, 4.0],
  "noise_source_ablation": true
}
```

Edit this one JSON file when you want to change the experiment.

## Realistic photonic nonidealities

The simulator includes:

- DAC quantization
- finite MZM extinction ratio
- MZM static bias error
- MZM random phase noise
- insertion loss
- shot noise
- laser RIN
- receiver thermal-current noise
- kTC reset noise
- charge integration
- ADC clipping and quantization

`noise_scale` is an **abstract random-noise sensitivity multiplier**. It multiplies the random-noise **standard deviation** of MZM phase noise and the combined receiver charge noise:

- `noise_scale = 0`: random noise off; DAC/ADC quantization and deterministic device nonidealities remain
- `noise_scale = 1`: baseline hardware values
- `noise_scale = 2`: 2x random-noise standard deviation (4x variance)

Default hardware values are starting assumptions unless independently measured/cited. Do not present them as measured properties of a real chip unless you have a source.

## What each sensitivity study changes

- ADC sweep changes **only `adc_bits`**.
- DAC sweep changes **only `dac_bits`**.
- Noise-scale sweep changes **only the random-noise standard-deviation multiplier**.
- Noise-source ablation enables `shot`, `rin`, `thermal`, `kTC`, or `MZM phase` individually, with `none` and `all` reference cases.

`hardware_eval_repeats` repeats stochastic photonic inference and reports mean/std loss. For publication, use several repeats rather than one stochastic pass.

## Fine N sweep

`all_in_one_paper.json` uses the fine-N model family with fixed `n_layer=4`, fixed `n_head=4`, and increasing `d_model`, so model size N and photonic dot-product length L change in a controlled way.

## Standalone scripts (optional)

The old two-stage interface is still available for debugging or rerunning sensitivity without retraining:

```powershell
python run_experiment.py --config configs/demo.json
python run_sensitivity.py --run runs\<timestamp> --config configs\sensitivity_demo.json
```

For normal use, `run_all.py` is recommended.

## Notes

- Use the **same corpus** throughout one run. `run_all.py` handles this automatically.
- `all_in_one_demo.json` and `all_in_one_fine_quick.json` are software checks, not paper conclusions.
- The ByteTokenizer is intentionally simple. For a language-model scaling study closer to modern LLM practice, replace it later with a fixed BPE/SentencePiece tokenizer and a controlled public corpus.
