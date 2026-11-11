from pathlib import Path
import torch


class ByteTokenizer:
    vocab_size = 256

    @staticmethod
    def encode(text):
        return torch.tensor(
            list(text.encode('utf-8', errors='replace')),
            dtype=torch.long,
        )


def demo_corpus():
    p = (
        'Photonic computing maps matrix multiplication onto physical signal propagation. '
        'Transformers repeatedly apply linear projections, attention, and feed forward layers. '
        'This demo corpus only verifies that the research pipeline runs end to end. '
    )
    return p * 800


def load_tokens(text_path=None):
    """Load one UTF-8 text file as byte tokens.

    If ``text_path`` is None, use the small built-in demo corpus. This function
    is kept for backward compatibility with the original single-corpus mode.
    """
    if text_path is None:
        text = demo_corpus()
    else:
        text = Path(text_path).read_text(
            encoding='utf-8',
            errors='replace',
        )
    return ByteTokenizer.encode(text)


def split_tokens(tokens):
    """Original 90/5/5 split used for a single corpus file."""
    n = len(tokens)
    a = int(0.90 * n)
    b = int(0.95 * n)
    return tokens[:a], tokens[a:b], tokens[b:]


def load_official_splits(data_dir):
    """Load official train/validation/test text files without re-splitting.

    Expected directory layout::

        data_dir/
            train.txt
            validation.txt
            test.txt

    Each file is independently UTF-8 byte-tokenized. No 90/5/5 split is
    applied, so the source dataset's official split boundaries are preserved.
    """
    data_dir = Path(data_dir)
    paths = {
        'train': data_dir / 'train.txt',
        'validation': data_dir / 'validation.txt',
        'test': data_dir / 'test.txt',
    }

    missing = [str(p) for p in paths.values() if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            'Official-split mode requires train.txt, validation.txt, and '
            f'test.txt. Missing: {missing}'
        )

    train = load_tokens(paths['train'])
    val = load_tokens(paths['validation'])
    test = load_tokens(paths['test'])
    return train, val, test


def load_data_splits(text_path=None, data_dir=None):
    """Load data in either backward-compatible or official-split mode.

    Modes
    -----
    1. ``data_dir`` is provided:
       use data_dir/train.txt, validation.txt, test.txt exactly as supplied.
       No re-splitting is performed.

    2. ``text_path`` is provided:
       load one UTF-8 corpus and apply the original 90/5/5 split.

    3. neither is provided:
       use the built-in demo corpus and apply the original 90/5/5 split.
    """
    if data_dir is not None and text_path is not None:
        raise ValueError(
            'Use either data_dir (official splits) or text_path '
            '(single corpus), not both.'
        )

    if data_dir is not None:
        return load_official_splits(data_dir)

    tokens = load_tokens(text_path)
    return split_tokens(tokens)


def dataset_info(train, val, test, text_path=None, data_dir=None):
    """Return a JSON-serializable summary of the exact dataset split used."""
    if data_dir is not None:
        mode = 'official_splits'
        source = str(Path(data_dir))
        paths = {
            'train': str(Path(data_dir) / 'train.txt'),
            'validation': str(Path(data_dir) / 'validation.txt'),
            'test': str(Path(data_dir) / 'test.txt'),
        }
    elif text_path is not None:
        mode = 'single_corpus_90_5_5'
        source = str(Path(text_path))
        paths = {'corpus': source}
    else:
        mode = 'built_in_demo_90_5_5'
        source = 'built_in_demo_corpus'
        paths = {}

    return {
        'mode': mode,
        'source': source,
        'paths': paths,
        'tokenizer': 'byte_level_utf8',
        'vocab_size': ByteTokenizer.vocab_size,
        'train_tokens_D': int(len(train)),
        'validation_tokens': int(len(val)),
        'test_tokens': int(len(test)),
        'total_tokens': int(len(train) + len(val) + len(test)),
        'official_split_boundaries_preserved': bool(data_dir is not None),
    }


def random_batch(data, batch_size, block_size, device):
    m = len(data) - block_size - 1
    if m <= 0:
        raise ValueError(
            f'Dataset split is too short for block_size={block_size}: '
            f'{len(data)} tokens.'
        )
    idx = torch.randint(0, m, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in idx])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in idx])
    return x.to(device), y.to(device)


def sequential_batches(data, batch_size, block_size, device, max_batches):
    starts = list(
        range(0, len(data) - block_size - 1, block_size)
    )[:max_batches * batch_size]

    for j in range(0, len(starts), batch_size):
        ss = starts[j:j + batch_size]
        if not ss:
            break
        x = torch.stack([data[i:i + block_size] for i in ss])
        y = torch.stack([data[i + 1:i + block_size + 1] for i in ss])
        yield x.to(device), y.to(device)
