# Digital vs Realistic Photonic Model-Shape Study

This add-on generates three absolute test-loss figures:

1. `ffn_ratio_vs_loss.png`
   - x = d_ff / d_model
   - y = test cross-entropy loss
   - Digital vs Realistic Photonic

2. `aspect_ratio_vs_loss.png`
   - x = d_model / n_layer
   - y = test cross-entropy loss
   - Digital vs Realistic Photonic

3. `head_dimension_vs_loss.png`
   - x = d_head = d_model / n_head
   - y = test cross-entropy loss
   - Digital vs Realistic Photonic

It also generates:
- `shape_study_digital_vs_photonic_3panel.png`
- `ffn_ratio.csv`
- `aspect_ratio.csv`
- `head_dimension.csv`
- `shape_study_all.csv`

The CSVs include:
- N_total
- N_nonembedding
- digital_loss
- photonic_loss_mean/std
- delta_loss = photonic - digital
- photonic_penalty_percent
- Nb_system_mean/std

## Installation

Copy these files into the project:

- `run_shape_study.py` -> project root
- `models/transformer.py` -> replace the old `models/transformer.py`
- `configs/shape_study.json` -> project `configs/`

The patched `ModelSpec` is backward compatible. Old model specs without `d_ff`
continue to use `d_ff = 4 * d_model`.

## Run with official WikiText-103 splits

```powershell
python run_shape_study.py `
  --config "configs\all_in_one_paper.json" `
  --shape-config "configs\shape_study.json" `
  --data-dir "data\wikitext103"
```

Use the same training config that you use for your main S01-S10 experiments.
If that config currently has `"train_steps": 10000`, the shape models also
train for 10000 steps and the best validation checkpoint is restored before
test evaluation.

## Experimental design

The FFN-ratio and aspect-ratio grids are chosen so that non-embedding parameter
counts remain roughly near the same target scale. The head-dimension sweep keeps
d_model, n_layer and d_ff fixed, so its parameter count is essentially constant.

For the paper, report the exact `N_nonembedding` values from the CSV rather than
claiming that all parameter counts are mathematically identical.
