#!/usr/bin/env python3
"""Download WikiText-103 raw and save the official splits as UTF-8 text."""

from pathlib import Path
from datasets import load_dataset


OUT = Path('data/wikitext103')
OUT.mkdir(parents=True, exist_ok=True)

print('Loading Salesforce/wikitext: wikitext-103-raw-v1 ...')
ds = load_dataset(
    'Salesforce/wikitext',
    'wikitext-103-raw-v1',
)

for split in ['train', 'validation', 'test']:
    path = OUT / f'{split}.txt'
    with path.open('w', encoding='utf-8', newline='\n') as f:
        for row in ds[split]:
            # Preserve the official split; only serialize rows to plain text.
            f.write(row['text'])
            f.write('\n')
    print(f'Saved {split}: {path}')

print('Done.')
print('Use: python run_all.py --config <config.json> --data-dir data/wikitext103')
