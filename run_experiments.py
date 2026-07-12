"""
run_experiments.py

This is where the ATTENTION PLACEMENT COMPARISON actually happens.
It trains the SAME CNNLSTMAttention model (from model_base.py) three
times -- once per attention position -- keeping everything else fixed
(same dataset, same seed, same hyperparameters). This is the controlled
experiment your paper reports in the Results table.

Usage:
    python run_experiments.py --dataset agnews --seeds 1 2 3
"""

import argparse
import random
import json
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import time

from model_base import CNNLSTMAttention
from data_loader import load_dataset_splits

ATTENTION_POSITIONS = ["before_cnn", "between", "after_lstm"]


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)


def evaluate(model, data_loader):
    model.eval()
    correct = 0
    total = 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in data_loader:
            logits = model(x)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
            all_preds.extend(preds.tolist())
            all_labels.extend(y.tolist())
    accuracy = correct / total
    return accuracy, all_preds, all_labels


def train_one_variant(attention_position, dataset_name, seed,
                       train_loader, test_loader, vocab_size, num_classes,
                       epochs=3):
    """Train ONE model variant and return its real metrics.
    Everything (data, hyperparameters, seed) is identical across variants
    except attention_position -- that is the whole point."""

    set_seed(seed)

    model = CNNLSTMAttention(
        vocab_size=vocab_size,
        embed_dim=128,
        cnn_channels=64,
        lstm_hidden=128,
        num_classes=num_classes,
        attention_position=attention_position,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    epoch_losses = []  # diagnostic safety net -- not plotted, just saved for debugging if needed

    start_time = time.time()
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        num_batches = 0
        for x, y in train_loader:
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            num_batches += 1
        avg_epoch_loss = total_loss / num_batches
        epoch_losses.append(avg_epoch_loss)
    train_time = time.time() - start_time

    inf_start = time.time()
    accuracy, preds, labels = evaluate(model, test_loader)
    inf_time = time.time() - inf_start

    num_params = sum(p.numel() for p in model.parameters())

    return {
        "attention_position": attention_position,
        "dataset": dataset_name,
        "seed": seed,
        "test_accuracy": accuracy,
        "train_time_sec": round(train_time, 2),
        "inference_time_sec": round(inf_time, 4),
        "num_parameters": num_params,
        "epoch_losses": [round(l, 4) for l in epoch_losses],
        "predictions": preds,
        "labels": labels,
    }


def run_all(dataset_name, seeds, epochs=3, tag=None, limit=None):
    print(f"Loading dataset '{dataset_name}'...")
    train_loader, test_loader, vocab_size, num_classes = load_dataset_splits(dataset_name, limit=limit)

    results = []
    for position in ATTENTION_POSITIONS:
        for seed in seeds:
            print(f"Training variant='{position}'  dataset='{dataset_name}'  seed={seed}")
            result = train_one_variant(position, dataset_name, seed,
                                        train_loader, test_loader,
                                        vocab_size, num_classes, epochs=epochs)
            print(f"  -> accuracy={result['test_accuracy']:.4f}  "
                  f"train_time={result['train_time_sec']}s")
            results.append(result)

    os.makedirs("results", exist_ok=True)
    # filename always includes the seed range, plus an optional tag,
    # so a small test run (e.g. seeds 1 2) never overwrites a full
    # final run (e.g. seeds 1 2 3 4 5) on the same dataset
    seed_str = "-".join(str(s) for s in seeds)
    suffix = f"_{tag}" if tag else ""
    out_path = f"results/{dataset_name}_seeds{seed_str}{suffix}_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {len(results)} results to {out_path}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="agnews",
                         choices=["agnews", "imdb", "arxiv"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--tag", type=str, default=None,
                         help="optional label to add to the results filename, e.g. 'test' or 'final'")
    parser.add_argument("--limit", type=int, default=None,
                         help="use only this many training examples, for FAST sanity checks only "
                              "(do not use --limit for your real reported results)")
    args = parser.parse_args()

    run_all(args.dataset, args.seeds, epochs=args.epochs, tag=args.tag, limit=args.limit)