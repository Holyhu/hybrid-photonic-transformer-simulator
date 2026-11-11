# Official WikiText split update

Replace these files in your existing project:

- `data/text.py`
- `run_experiment.py`
- `run_sensitivity.py`
- `run_all.py`

`run_sensitivity.py` is based on the current Nb-system + Nb-scaling version.

## 1. Download WikiText-103 raw

Install the downloader dependency once:

```powershell
pip install datasets
```

Then:

```powershell
python download_wikitext103.py
```

It creates:

```text
data/wikitext103/
  train.txt
  validation.txt
  test.txt
```

These correspond to the dataset's official train / validation / test splits.

## 2. Run the complete experiment

```powershell
python run_all.py `
  --config "configs\all_in_one_paper.json" `
  --data-dir "data\wikitext103"
```

The program does **not** call the old 90/5/5 splitter in this mode.

## 3. Run only sensitivity on existing checkpoints

Only do this if those checkpoints were trained with the same WikiText split files:

```powershell
python run_sensitivity.py `
  --run "runs\YOUR_RUN" `
  --config "configs\sensitivity_with_Nb_scaling.json" `
  --data-dir "data\wikitext103"
```

## Backward compatibility

The old command still works:

```powershell
python run_all.py --config "configs\all_in_one_paper.json" --text "corpus.txt"
```

In `--text` mode the original 90/5/5 split is still used.

## Dataset metadata

Each new run writes `dataset_info.json`, including:

- `train_tokens_D`
- `validation_tokens`
- `test_tokens`
- data mode and source paths
- whether official split boundaries were preserved

For the thesis, `train_tokens_D` is the byte-token dataset size D used by this program.
