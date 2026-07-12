"""
investigate_instability.py

Uses the epoch_losses saved in a results JSON file to diagnose WHY
before_cnn / between struggled on a given dataset (e.g. ArXiv, IMDB).

Prints, per position, the per-seed loss trajectory across epochs, so we
can see whether the model:
    (a) never started learning (loss stays flat near ln(num_classes)), or
    (b) learned partially then plateaued, or
    (c) learned well (loss decreases steadily) -- like after_lstm should.

Also flags which specific seeds looks "stuck" (loss barely moves).

Usage:
    python investigate_instability.py --file results/arxiv_seeds1-2-3-4-5_final_results.json
"""

import argparse
import json
import math
from collections import defaultdict


def load_results(path):
    with open(path, "r") as f:
        return json.load(f)


def group_by_position(results):
    groups = defaultdict(list)
    for run in results:
        groups[run["attention_position"]].append(run)
    return groups


def random_baseline_loss(num_classes):
    # cross-entropy loss of a model that just guesses uniformly at random
    return math.log(num_classes)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, required=True)
    args = parser.parse_args()

    results = load_results(args.file)
    dataset_name = results[0]["dataset"]
    groups = group_by_position(results)

    if "epoch_losses" not in results[0]:
        print("This results file doesn't have epoch_losses saved "
              "(likely an older run, before the diagnostic logging was added). "
              "Nothing to investigate here.")
        exit()

    # infer num_classes from confusion-matrix-worthy label range, for the baseline reference
    all_labels = set()
    for r in results:
        all_labels.update(r["labels"])
    num_classes = len(all_labels)
    baseline = random_baseline_loss(num_classes)

    print(f"\nDataset: {dataset_name}  |  num_classes={num_classes}  "
          f"|  random-guessing loss ~= {baseline:.3f}\n")

    for position in ["before_cnn", "between", "after_lstm"]:
        if position not in groups:
            continue
        print("=" * 70)
        print(f"POSITION: {position}")
        print("=" * 70)
        for run in sorted(groups[position], key=lambda r: r["seed"]):
            losses = run["epoch_losses"]
            acc = run["test_accuracy"]
            first, last = losses[0], losses[-1]
            drop = first - last
            status = ""
            if last > baseline * 0.9:
                status = "STUCK near random-guessing loss (never really learned)"
            elif drop < 0.05:
                status = "FLAT (loss barely moved across epochs)"
            else:
                status = "learned normally (loss decreased meaningfully)"
            print(f"  seed={run['seed']}  acc={acc:.3f}  "
                  f"losses={[round(l,3) for l in losses]}  -> {status}")
        print()