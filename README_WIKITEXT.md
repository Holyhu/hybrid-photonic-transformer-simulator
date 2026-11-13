# WikiText-103 setup

The thesis experiments use `Salesforce/wikitext`, configuration
`wikitext-103-raw-v1`, preserving the official train/validation/test splits.

## Download

```powershell
python -m pip install -r requirements.txt
python download_wikitext103.py
```

This creates:

```text
data/wikitext103/
  train.txt
  validation.txt
  test.txt
```

## Run

```powershell
python run_all.py `
  --config "configs\all_in_one_paper.json" `
  --data-dir "data\wikitext103"
```

No additional 90/5/5 split is performed in `--data-dir` mode.

The legacy `--text` mode remains available and applies the older 90/5/5
single-corpus split.

Each run records `dataset_info.json`, including byte-token counts and whether
official split boundaries were preserved.
