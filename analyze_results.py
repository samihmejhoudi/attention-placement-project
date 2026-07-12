"""
analyze_results.py

Reads a results JSON file (produced by run_experiments.py) and computes:
    - Accuracy, Precision, Recall, F1-score (per run)
    - Mean +/- Standard Deviation across seeds (per attention position)
    - Confusion Matrix (per attention position, aggregated across seeds)
    - Statistical significance tests (paired t-test between positions,
      using paired-seed accuracy differences)
    - Accuracy gain table (pairwise percentage-point differences)
    - Parameter count and timing summary

Usage:
    python analyze_results.py --file results/agnews_seeds1-2-3-4-5_final_results.json
"""

import argparse
import json
from collections import defaultdict

import numpy as np
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from scipy import stats


def load_results(path):
    with open(path, "r") as f:
        return json.load(f)


def per_run_metrics(run):
    """Compute precision/recall/F1 for a single run from its raw predictions."""
    preds = run["predictions"]
    labels = run["labels"]
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )
    return {
        "accuracy": run["test_accuracy"],
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def group_by_position(results):
    groups = defaultdict(list)
    for run in results:
        groups[run["attention_position"]].append(run)
    return groups


def summarize(groups):
    """Mean +/- std for accuracy/precision/recall/F1/timing/params per position."""
    summary = {}
    for position, runs in groups.items():
        metrics_per_run = [per_run_metrics(r) for r in runs]
        acc = [m["accuracy"] for m in metrics_per_run]
        prec = [m["precision"] for m in metrics_per_run]
        rec = [m["recall"] for m in metrics_per_run]
        f1 = [m["f1"] for m in metrics_per_run]
        train_time = [r["train_time_sec"] for r in runs]
        params = runs[0]["num_parameters"]  # same for all seeds of a position

        summary[position] = {
            "n_seeds": len(runs),
            "accuracy_mean": np.mean(acc), "accuracy_std": np.std(acc),
            "precision_mean": np.mean(prec), "precision_std": np.std(prec),
            "recall_mean": np.mean(rec), "recall_std": np.std(rec),
            "f1_mean": np.mean(f1), "f1_std": np.std(f1),
            "train_time_mean_sec": np.mean(train_time),
            "num_parameters": params,
            "raw_accuracies": acc,  # kept for paired significance testing
        }
    return summary


def significance_tests(summary):
    """Paired t-test between each pair of positions, using per-seed accuracy.
    Assumes runs were saved in matching seed order for each position."""
    positions = list(summary.keys())
    results = {}
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            p1, p2 = positions[i], positions[j]
            acc1 = summary[p1]["raw_accuracies"]
            acc2 = summary[p2]["raw_accuracies"]
            if len(acc1) == len(acc2) and len(acc1) > 1:
                t_stat, p_value = stats.ttest_rel(acc1, acc2)
                results[f"{p1} vs {p2}"] = {"t_stat": t_stat, "p_value": p_value}
    return results


def accuracy_gain_table(summary):
    """Pairwise accuracy gain (in percentage points) between all positions."""
    positions = list(summary.keys())
    gains = []
    for i in range(len(positions)):
        for j in range(len(positions)):
            if i != j:
                p1, p2 = positions[i], positions[j]
                gain = (summary[p1]["accuracy_mean"] - summary[p2]["accuracy_mean"]) * 100
                gains.append((f"{p1} vs {p2}", gain))
    return gains


def aggregate_confusion_matrix(groups):
    """One confusion matrix per position, summed across all seeds."""
    matrices = {}
    for position, runs in groups.items():
        all_preds, all_labels = [], []
        for r in runs:
            all_preds.extend(r["predictions"])
            all_labels.extend(r["labels"])
        matrices[position] = confusion_matrix(all_labels, all_preds)
    return matrices


def print_report(summary, sig_tests, matrices, gains):
    print("\n" + "=" * 70)
    print("SUMMARY TABLE (mean +/- std across seeds)")
    print("=" * 70)
    header = f"{'Position':<14}{'Accuracy':<16}{'Precision':<16}{'Recall':<16}{'F1':<16}{'Params':<10}"
    print(header)
    print("-" * len(header))
    for position, s in summary.items():
        print(f"{position:<14}"
              f"{s['accuracy_mean']*100:.2f}+/-{s['accuracy_std']*100:.2f}%   "
              f"{s['precision_mean']*100:.2f}+/-{s['precision_std']*100:.2f}%   "
              f"{s['recall_mean']*100:.2f}+/-{s['recall_std']*100:.2f}%   "
              f"{s['f1_mean']*100:.2f}+/-{s['f1_std']*100:.2f}%   "
              f"{s['num_parameters']:,}")

    print("\n" + "=" * 70)
    print("ACCURACY GAIN TABLE (percentage points)")
    print("=" * 70)
    print(f"{'Comparison':<30}{'Accuracy Gain':<15}")
    print("-" * 45)
    for comparison, gain in gains:
        sign = "+" if gain >= 0 else ""
        print(f"{comparison:<30}{sign}{gain:.2f}%")

    print("\n" + "=" * 70)
    print("STATISTICAL SIGNIFICANCE (paired t-test on per-seed accuracy)")
    print("=" * 70)
    for comparison, result in sig_tests.items():
        sig_marker = "SIGNIFICANT (p<0.05)" if result["p_value"] < 0.05 else "not significant"
        print(f"{comparison}: t={result['t_stat']:.3f}, p={result['p_value']:.4f}  [{sig_marker}]")

    print("\n" + "=" * 70)
    print("CONFUSION MATRICES (summed across all seeds)")
    print("=" * 70)
    for position, matrix in matrices.items():
        print(f"\n{position}:")
        print(matrix)

    print("\n" + "=" * 70)
    print("TRAINING TIME SUMMARY")
    print("=" * 70)
    for position, s in summary.items():
        print(f"{position:<14} avg train time: {s['train_time_mean_sec']:.1f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, required=True,
                         help="path to a results JSON file from run_experiments.py")
    args = parser.parse_args()

    results = load_results(args.file)
    groups = group_by_position(results)
    summary = summarize(groups)
    sig_tests = significance_tests(summary)
    matrices = aggregate_confusion_matrix(groups)
    gains = accuracy_gain_table(summary)

    print_report(summary, sig_tests, matrices, gains)

    # also save a clean JSON summary for easy reuse in the paper
    out_path = args.file.replace(".json", "_summary.json")
    clean_summary = {
        pos: {k: v for k, v in s.items() if k != "raw_accuracies"}
        for pos, s in summary.items()
    }
    with open(out_path, "w") as f:
        json.dump({
            "summary": clean_summary,
            "significance_tests": {k: {kk: float(vv) for kk, vv in v.items()}
                                    for k, v in sig_tests.items()},
        }, f, indent=2)
    print(f"\nSaved clean summary to {out_path}")