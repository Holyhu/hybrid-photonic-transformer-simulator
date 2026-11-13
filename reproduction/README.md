# Reproduction record

Use this directory to freeze the exact artifacts used for the final thesis figures.

Recommended contents:

- `all_in_one_paper.json` — exact final configuration
- `sensitivity_final.json` — exact sensitivity configuration
- final figure-source CSV files
- `COMMIT.txt` — Git commit hash used for the final run
- `ENVIRONMENT.txt` — Python/PyTorch/ROCm environment summary

Do not place large checkpoints or the WikiText-103 dataset here unless you explicitly intend to publish them.

Example:

```powershell
git rev-parse HEAD > reproduction\COMMIT.txt
```
