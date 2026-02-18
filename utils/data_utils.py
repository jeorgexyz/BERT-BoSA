import re
from collections import Counter
from functools import partial

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

# Special-token indices (must match SPECIALS order below)
PAD_IDX = 0
UNK_IDX = 1
CLS_IDX = 2
SEP_IDX = 3
SPECIALS = ['<pad>', '<unk>', '<cls>', '<sep>']


def basic_tokenizer(text):
    """Lowercase + split on word boundaries — equivalent to torchtext basic_english."""
    return re.findall(r'\b\w+\b', text.lower())


class Vocab:
    def __init__(self, token2idx):
        self.token2idx = token2idx

    def __len__(self):
        return len(self.token2idx)

    def __call__(self, tokens):
        return [self.token2idx.get(t, UNK_IDX) for t in tokens]

    def __getitem__(self, token):
        return self.token2idx.get(token, UNK_IDX)


def build_vocab(texts, max_tokens=30000):
    counter = Counter()
    for text in texts:
        counter.update(basic_tokenizer(text))

    token2idx = {tok: i for i, tok in enumerate(SPECIALS)}
    for token, _ in counter.most_common(max_tokens - len(SPECIALS)):
        if token not in token2idx:
            token2idx[token] = len(token2idx)

    return Vocab(token2idx)


def _load_ag_news():
    """Download AG_NEWS via HuggingFace datasets. Returns lists of (label_1to4, text)."""
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "Run: pip install datasets\n"
            "(torchtext is incompatible with your PyTorch version)"
        )
    ds = load_dataset('ag_news')
    # HuggingFace ag_news labels are 0–3; shift to 1–4 to stay consistent
    train_data = [(item['label'] + 1, item['text']) for item in ds['train']]
    test_data  = [(item['label'] + 1, item['text']) for item in ds['test']]
    return train_data, test_data


def _encode_text(text, vocab, max_len):
    tokens = ['<cls>'] + basic_tokenizer(text)[: max_len - 2] + ['<sep>']
    return torch.tensor(vocab(tokens), dtype=torch.long)


def _collate_batch(batch, vocab, max_len):
    labels, texts = zip(*batch)
    label_tensor = torch.tensor([l - 1 for l in labels], dtype=torch.long)  # shift to 0–3
    encoded = [_encode_text(t, vocab, max_len) for t in texts]
    padded = pad_sequence(encoded, batch_first=True, padding_value=PAD_IDX)
    segments = torch.zeros_like(padded)
    return padded, segments, label_tensor


def get_dataloaders(config):
    batch_size = config['batch_size']
    max_len    = config['max_len']
    max_tokens = config.get('vocab_size', 30000)

    print("Downloading AG_NEWS dataset...")
    train_data, test_data = _load_ag_news()

    print("Building vocabulary...")
    vocab = build_vocab([text for _, text in train_data], max_tokens=max_tokens)

    collate_fn = partial(_collate_batch, vocab=vocab, max_len=max_len)

    train_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True,  collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_data,  batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )
    return train_loader, test_loader, vocab
