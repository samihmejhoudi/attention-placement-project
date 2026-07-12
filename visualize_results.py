"""
visualize_results.py

Generates real charts from a results JSON file (produced by
run_experiments.py):
    - Bar chart: accuracy per attention position, with std-dev error bars
    - Confusion matrix heatmaps: one per position
    - Precision/Recall/F1 grouped bar chart
    - Training time comparison
    - Parameter count comparison
    - Statistical significance table
    - Accuracy gain chart

Figures are saved in a subfolder per dataset to keep things organized:
    results/figures/agnews/...
    results/figures/imdb/...
    results/figures/arxiv/...

Usage:
    python visualize_results.py --file results/agnews_seeds1-2-3-4-5_final_results.json
"""

import argparse
import json
import os
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from scipy import stats

# Human-readable labels for the attention positions, used in all figures
NICE_LABELS = {
    "before_cnn": "Before CNN",
    "between": "Between CNN and LSTM",
    "after_lstm": "After LSTM",
}

DPI = 300


def nice(position):
    return NICE_LABELS.get(position, position)


def load_results(path):
    with open(path, "r") as f:
        return json.load(f)


def group_by_position(results):
    groups = defaultdict(list)
    for run in results:
        groups[run["attention_position"]].append(run)
    return groups


def compute_prf(run):
    precision, recall, f1, _ = precision_recall_fscore_support(
        run["labels"], run["predictions"], average="macro", zero_division=0
    )
    return precision, recall, f1


def plot_accuracy_bar(groups, dataset_name, out_dir):
    positions = list(groups.keys())
    labels = [nice(p) for p in positions]
    means = [np.mean([r["test_accuracy"] for r in groups[p]]) * 100 for p in positions]
    stds = [np.std([r["test_accuracy"] for r in groups[p]]) * 100 for p in positions]

    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(labels, means, yerr=stds, capsize=8, color=["#4C72B0", "#55A868", "#C44E52"])
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title(f"Accuracy by Attention Position — {dataset_name}")
    ax.set_ylim(min(means) - 2, max(means) + 2)
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, mean + 0.15, f"{mean:.2f}%",
                ha="center", va="bottom", fontsize=10)
    plt.tight_layout()
    path = os.path.join(out_dir, f"{dataset_name}_accuracy_bar.png")
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


def plot_prf_grouped_bar(groups, dataset_name, out_dir):
    positions = list(groups.keys())
    labels_x = [nice(p) for p in positions]
    prec_means, rec_means, f1_means = [], [], []
    prec_stds, rec_stds, f1_stds = [], [], []

    for p in positions:
        precs, recs, f1s = [], [], []
        for r in groups[p]:
            prec, rec, f1 = compute_prf(r)
            precs.append(prec * 100)
            recs.append(rec * 100)
            f1s.append(f1 * 100)
        prec_means.append(np.mean(precs)); prec_stds.append(np.std(precs))
        rec_means.append(np.mean(recs)); rec_stds.append(np.std(recs))
        f1_means.append(np.mean(f1s)); f1_stds.append(np.std(f1s))

    x = np.arange(len(positions))
    width = 0.25

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width, prec_means, width, yerr=prec_stds, capsize=5, label="Precision", color="#4C72B0")
    ax.bar(x, rec_means, width, yerr=rec_stds, capsize=5, label="Recall", color="#55A868")
    ax.bar(x + width, f1_means, width, yerr=f1_stds, capsize=5, label="F1-score", color="#C44E52")
    ax.set_xticks(x)
    ax.set_xticklabels(labels_x)
    ax.set_ylabel("Score (%)")
    ax.set_title(f"Precision / Recall / F1 by Attention Position — {dataset_name}")
    ax.set_ylim(min(prec_means + rec_means + f1_means) - 3, max(prec_means + rec_means + f1_means) + 3)
    ax.legend()
    plt.tight_layout()
    path = os.path.join(out_dir, f"{dataset_name}_prf_grouped_bar.png")
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


def plot_training_time(groups, dataset_name, out_dir):
    positions = list(groups.keys())
    labels = [nice(p) for p in positions]
    means = [np.mean([r["train_time_sec"] for r in groups[p]]) for p in positions]
    stds = [np.std([r["train_time_sec"] for r in groups[p]]) for p in positions]

    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(labels, means, yerr=stds, capsize=8, color=["#8172B2", "#CCB974", "#64B5CD"])
    ax.set_ylabel("Training Time (seconds)")
    ax.set_title(f"Training Time by Attention Position — {dataset_name}")
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, mean + max(means) * 0.01,
                f"{mean:.0f}s", ha="center", va="bottom", fontsize=10)
    plt.tight_layout()
    path = os.path.join(out_dir, f"{dataset_name}_training_time.png")
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


def plot_parameter_comparison(groups, dataset_name, out_dir):
    """Bar chart showing parameter counts are nearly identical across
    positions -- supports the claim that the comparison is fair (the
    accuracy differences come from WHERE attention sits, not from extra
    model capacity)."""
    positions = list(groups.keys())
    labels = [nice(p) for p in positions]
    params = [groups[p][0]["num_parameters"] for p in positions]

    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(labels, params, color=["#4C72B0", "#55A868", "#C44E52"])
    ax.set_ylabel("Number of Parameters")
    ax.set_title(f"Parameter Count by Attention Position — {dataset_name}")
    pad = max(params) * 0.0005
    ax.set_ylim(min(params) - pad, max(params) + pad)
    for bar, p in zip(bars, params):
        ax.text(bar.get_x() + bar.get_width() / 2, p + pad * 0.1, f"{p:,}",
                ha="center", va="bottom", fontsize=10)
    plt.tight_layout()
    path = os.path.join(out_dir, f"{dataset_name}_parameter_comparison.png")
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


def plot_significance_table(groups, dataset_name, out_dir):
    positions = list(groups.keys())
    accs = {p: [r["test_accuracy"] for r in groups[p]] for p in positions}

    rows = []
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            p1, p2 = positions[i], positions[j]
            if len(accs[p1]) == len(accs[p2]) and len(accs[p1]) > 1:
                t_stat, p_val = stats.ttest_rel(accs[p1], accs[p2])
                sig = "Yes (p<0.05)" if p_val < 0.05 else "No"
                rows.append([f"{nice(p1)} vs {nice(p2)}", f"{t_stat:.3f}", f"{p_val:.4f}", sig])

    fig, ax = plt.subplots(figsize=(9, 1 + 0.5 * len(rows)))
    ax.axis("off")
    table = ax.table(
        cellText=rows,
        colLabels=["Comparison", "t-statistic", "p-value", "Significant?"],
        cellLoc="center", loc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.8)
    ax.set_title(f"Statistical Significance — {dataset_name}", pad=20)
    plt.tight_layout()
    path = os.path.join(out_dir, f"{dataset_name}_significance_table.png")
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


def plot_accuracy_gain_chart(groups, dataset_name, out_dir):
    positions = list(groups.keys())
    means = {p: np.mean([r["test_accuracy"] for r in groups[p]]) * 100 for p in positions}

    labels, gains = [], []
    for i in range(len(positions)):
        for j in range(len(positions)):
            if i != j:
                p1, p2 = positions[i], positions[j]
                labels.append(f"{nice(p1)}\nvs\n{nice(p2)}")
                gains.append(means[p1] - means[p2])

    colors = ["#55A868" if g >= 0 else "#C44E52" for g in gains]
    fig, ax = plt.subplots(figsize=(11, 5))
    bars = ax.bar(labels, gains, color=colors)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Accuracy Gain (percentage points)")
    ax.set_title(f"Pairwise Accuracy Gain — {dataset_name}")
    for bar, gain in zip(bars, gains):
        sign = "+" if gain >= 0 else ""
        ax.text(bar.get_x() + bar.get_width() / 2, gain + (0.03 if gain >= 0 else -0.08),
                f"{sign}{gain:.2f}%", ha="center",
                va="bottom" if gain >= 0 else "top", fontsize=9)
    plt.tight_layout()
    path = os.path.join(out_dir, f"{dataset_name}_accuracy_gain_chart.png")
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


def plot_confusion_heatmaps(groups, dataset_name, out_dir):
    for position, runs in groups.items():
        all_preds, all_labels = [], []
        for r in runs:
            all_preds.extend(r["predictions"])
            all_labels.extend(r["labels"])
        cm = confusion_matrix(all_labels, all_preds)
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

        fig, ax = plt.subplots(figsize=(5, 5))
        im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
        ax.set_title(f"Confusion Matrix — {nice(position)} ({dataset_name})")
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color="white" if cm_norm[i, j] > 0.5 else "black", fontsize=9)
        fig.colorbar(im, ax=ax, label="Proportion")
        plt.tight_layout()
        path = os.path.join(out_dir, f"{dataset_name}_confusion_{position}.png")
        plt.savefig(path, dpi=DPI, bbox_inches="tight")
        plt.close()
        print(f"Saved {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, required=True,
                         help="path to a results JSON file from run_experiments.py")
    parser.add_argument("--out_dir", type=str, default="results/figures",
                         help="base folder; each dataset gets its own subfolder inside this")
    args = parser.parse_args()

    results = load_results(args.file)
    dataset_name = results[0]["dataset"]
    groups = group_by_position(results)

    # put each dataset's figures in its own subfolder to avoid clutter
    out_dir = os.path.join(args.out_dir, dataset_name)
    os.makedirs(out_dir, exist_ok=True)

    plot_accuracy_bar(groups, dataset_name, out_dir)
    plot_confusion_heatmaps(groups, dataset_name, out_dir)
    plot_prf_grouped_bar(groups, dataset_name, out_dir)
    plot_training_time(groups, dataset_name, out_dir)
    plot_parameter_comparison(groups, dataset_name, out_dir)
    plot_significance_table(groups, dataset_name, out_dir)
    plot_accuracy_gain_chart(groups, dataset_name, out_dir)

    print(f"\nAll charts saved to {out_dir}/")