"""
compare_datasets.py

Combines results across all 3 datasets (AG News = short, IMDB = medium,
ArXiv = long) into ONE comparison view. This directly answers the core
research question: "Does attention placement matter, and does the
effect depend on sequence length?"

Generates:
    - A combined accuracy-by-sequence-length line/bar chart (the single
      most important figure in the paper)
    - A combined summary table (printed + saved as JSON)

Usage:
    python compare_datasets.py \\
        --agnews results/agnews_seeds1-2-3-4-5_final_results.json \\
        --imdb results/imdb_seeds1-2-3-4-5_final_results.json \\
        --arxiv results/arxiv_seeds1-2-3-4-5_final_results.json
"""

import argparse
import json
import os
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

NICE_LABELS = {
    "before_cnn": "Before CNN",
    "between": "Between CNN and LSTM",
    "after_lstm": "After LSTM",
}

# approximate sequence lengths used for each dataset (matches data_loader.py max_len)
SEQ_LENGTHS = {
    "agnews": 40,
    "imdb": 200,
    "arxiv": 1000,
}

DATASET_DISPLAY_NAMES = {
    "agnews": "AG News (short)",
    "imdb": "IMDB (medium)",
    "arxiv": "ArXiv (long)",
}

DPI = 300


def nice(position):
    return NICE_LABELS.get(position, position)


def load_and_group(path):
    with open(path, "r") as f:
        results = json.load(f)
    groups = defaultdict(list)
    for run in results:
        groups[run["attention_position"]].append(run["test_accuracy"])
    dataset_name = results[0]["dataset"]
    return dataset_name, groups


def build_combined_summary(dataset_paths):
    """dataset_paths: dict like {'agnews': path, 'imdb': path, 'arxiv': path}"""
    summary = {}
    for key, path in dataset_paths.items():
        dataset_name, groups = load_and_group(path)
        summary[key] = {
            position: {
                "mean": float(np.mean(accs)) * 100,
                "std": float(np.std(accs)) * 100,
                "n_seeds": len(accs),
            }
            for position, accs in groups.items()
        }
    return summary


def plot_cross_dataset_comparison(summary, out_dir):
    """The key figure: accuracy vs sequence length, one line per attention position."""
    positions = ["before_cnn", "between", "after_lstm"]
    dataset_keys = [k for k in ["agnews", "imdb", "arxiv"] if k in summary]
    x_lengths = [SEQ_LENGTHS[k] for k in dataset_keys]
    x_labels = [DATASET_DISPLAY_NAMES[k] for k in dataset_keys]

    colors = {"before_cnn": "#4C72B0", "between": "#55A868", "after_lstm": "#C44E52"}
    markers = {"before_cnn": "o", "between": "s", "after_lstm": "^"}

    fig, ax = plt.subplots(figsize=(9, 6))
    for position in positions:
        means = [summary[k][position]["mean"] for k in dataset_keys if position in summary[k]]
        stds = [summary[k][position]["std"] for k in dataset_keys if position in summary[k]]
        xs = [SEQ_LENGTHS[k] for k in dataset_keys if position in summary[k]]
        ax.errorbar(xs, means, yerr=stds, label=nice(position),
                    color=colors[position], marker=markers[position],
                    markersize=9, linewidth=2, capsize=6)

    ax.set_xscale("log")
    ax.set_xticks(x_lengths)
    ax.set_xticklabels([f"{l}\n({lbl})" for l, lbl in zip(x_lengths, x_labels)])
    ax.set_xlabel("Sequence Length (approx. max tokens)")
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title("Attention Placement Performance Across Sequence Lengths")
    ax.legend(title="Attention Position")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(out_dir, "cross_dataset_accuracy_by_sequence_length.png")
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


def plot_gap_widening_chart(summary, out_dir):
    """Shows the accuracy GAP (after_lstm minus the others) growing with sequence length."""
    dataset_keys = [k for k in ["agnews", "imdb", "arxiv"] if k in summary]
    x_lengths = [SEQ_LENGTHS[k] for k in dataset_keys]
    x_labels = [DATASET_DISPLAY_NAMES[k] for k in dataset_keys]

    gap_before = [summary[k]["after_lstm"]["mean"] - summary[k]["before_cnn"]["mean"] for k in dataset_keys]
    gap_between = [summary[k]["after_lstm"]["mean"] - summary[k]["between"]["mean"] for k in dataset_keys]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(x_lengths, gap_before, marker="o", markersize=9, linewidth=2,
            color="#4C72B0", label="After LSTM vs Before CNN")
    ax.plot(x_lengths, gap_between, marker="s", markersize=9, linewidth=2,
            color="#55A868", label="After LSTM vs Between CNN and LSTM")

    ax.set_xscale("log")
    ax.set_xticks(x_lengths)
    ax.set_xticklabels([f"{l}\n({lbl})" for l, lbl in zip(x_lengths, x_labels)])
    ax.set_xlabel("Sequence Length (approx. max tokens)")
    ax.set_ylabel("Accuracy Gap (percentage points)")
    ax.set_title("The Accuracy Gap Widens With Sequence Length")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(out_dir, "cross_dataset_gap_widening.png")
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


def print_combined_table(summary):
    dataset_keys = [k for k in ["agnews", "imdb", "arxiv"] if k in summary]
    print("\n" + "=" * 80)
    print("COMBINED SUMMARY -- ACCURACY (%) MEAN +/- STD, ACROSS ALL DATASETS")
    print("=" * 80)
    header = f"{'Position':<24}" + "".join(f"{DATASET_DISPLAY_NAMES[k]:<22}" for k in dataset_keys)
    print(header)
    print("-" * len(header))
    for position in ["before_cnn", "between", "after_lstm"]:
        row = f"{nice(position):<24}"
        for k in dataset_keys:
            if position in summary[k]:
                m = summary[k][position]["mean"]
                s = summary[k][position]["std"]
                row += f"{m:.2f}+/-{s:.2f}%       "
            else:
                row += f"{'N/A':<22}"
        print(row)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--agnews", type=str, default=None)
    parser.add_argument("--imdb", type=str, default=None)
    parser.add_argument("--arxiv", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default="results/figures/combined")
    args = parser.parse_args()

    dataset_paths = {}
    if args.agnews:
        dataset_paths["agnews"] = args.agnews
    if args.imdb:
        dataset_paths["imdb"] = args.imdb
    if args.arxiv:
        dataset_paths["arxiv"] = args.arxiv

    if len(dataset_paths) < 2:
        print("Provide at least 2 dataset result files to compare (e.g. --agnews ... --imdb ... --arxiv ...)")
        exit()

    os.makedirs(args.out_dir, exist_ok=True)

    summary = build_combined_summary(dataset_paths)
    print_combined_table(summary)
    plot_cross_dataset_comparison(summary, args.out_dir)
    plot_gap_widening_chart(summary, args.out_dir)

    out_json = os.path.join(args.out_dir, "combined_summary.json")
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved combined summary to {out_json}")
    print(f"All charts saved to {args.out_dir}/")