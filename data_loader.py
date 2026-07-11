"""
data_loader.py

Loads and prepares the three sequence-length datasets:
    - agnews : short sequences
    - imdb   : medium sequences
    - arxiv  : long sequences

Uses the Hugging Face `datasets` library (pip install datasets).
Tokenization is done with a simple whitespace + vocabulary approach
to keep the pipeline self-contained (no external tokenizer needed).
"""

import torch
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from datasets import load_dataset

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"


def build_vocab(texts, max_vocab_size=20000):
    counter = Counter()
    for text in texts:
        counter.update(text.lower().split())
    most_common = counter.most_common(max_vocab_size - 2)
    vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for i, (word, _) in enumerate(most_common, start=2):
        vocab[word] = i
    return vocab


def encode(text, vocab, max_len):
    tokens = text.lower().split()[:max_len]
    ids = [vocab.get(tok, vocab[UNK_TOKEN]) for tok in tokens]
    ids += [vocab[PAD_TOKEN]] * (max_len - len(ids))
    return ids


class TextClassificationDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        ids = encode(self.texts[idx], self.vocab, self.max_len)
        return torch.tensor(ids, dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)


DATASET_CONFIG = {
    "agnews": {"hf_name": "fancyzhx/ag_news",  "max_len": 40,  "num_classes": 4},
    "imdb":   {"hf_name": "stanfordnlp/imdb",  "max_len": 200, "num_classes": 2},
    "arxiv":  {"hf_name": "ccdv/arxiv-classification", "max_len": 1000, "num_classes": 11},
}


def load_dataset_splits(dataset_name, batch_size=32, max_vocab_size=20000, limit=None):
    if dataset_name not in DATASET_CONFIG:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    cfg = DATASET_CONFIG[dataset_name]
    raw = load_dataset(cfg["hf_name"])

    train_texts = raw["train"]["text"]
    train_labels = raw["train"]["label"]
    test_texts = raw["test"]["text"]
    test_labels = raw["test"]["label"]

    if limit is not None:
        train_texts = train_texts[:limit]
        train_labels = train_labels[:limit]
        test_texts = test_texts[:max(limit // 5, 50)]
        test_labels = test_labels[:max(limit // 5, 50)]

    vocab = build_vocab(train_texts, max_vocab_size=max_vocab_size)

    train_ds = TextClassificationDataset(train_texts, train_labels, vocab, cfg["max_len"])
    test_ds = TextClassificationDataset(test_texts, test_labels, vocab, cfg["max_len"])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, len(vocab), cfg["num_classes"]


if __name__ == "__main__":
    train_loader, test_loader, vocab_size, num_classes = load_dataset_splits("agnews", batch_size=8, limit=500)
    batch_x, batch_y = next(iter(train_loader))
    print("vocab_size:", vocab_size)
    print("num_classes:", num_classes)
    print("batch_x shape:", batch_x.shape)
    print("batch_y shape:", batch_y.shape)