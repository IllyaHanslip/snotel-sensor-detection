"""
Visualisation: compare all four detection methods side by side.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from stations_config import get_primary_station

# cur/sta = current/station, proc = processed, fig = figures
cur_sta = get_primary_station()
proc_dir = Path("data/processed") / cur_sta["triplet"].replace(":", "_")
fig_dir = Path("figures") / cur_sta["triplet"].replace(":", "_")

res_files = {  # res = results
    "Baseline\n(rolling median)": "baseline_filter_results.csv",
    "ARIMA\nresidual": "arima_model_results.csv",
    "Isolation\nForest": "isolation_forest_results.csv",
    "LSTM\nautoencoder": "lstm_autoencoder_results.csv",
}


def compute_metrics_from_file(filename):
    """Recompute precision, recall, and F1 from one method's saved
    results CSV."""
    df = pd.read_csv(proc_dir / filename)
    is_noise = df["is_noise"]
    pred_noise = df["predicted_noise"]

    # tp/fp/fn = true/false positive/negative
    tp = (is_noise & pred_noise).sum()
    fp = (~is_noise & pred_noise).sum()
    fn = (is_noise & ~pred_noise).sum()

    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0  # prec = precision
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * recall / (prec + recall) if (prec + recall) > 0 else 0.0

    return prec, recall, f1


def main():
    """Draw a bar chart comparing all four methods' precision, recall,
    and F1."""
    method_names = list(res_files.keys())
    precs, recalls, f1s = [], [], []

    for name, filename in res_files.items():
        path = proc_dir / filename
        if not path.exists():
            raise FileNotFoundError(
                f"{filename} not found - run the matching model script first "
                f"(e.g. baseline_filter.py) before this one."
            )
        prec, recall, f1 = compute_metrics_from_file(filename)
        precs.append(prec)
        recalls.append(recall)
        f1s.append(f1)
        print(f"{name.replace(chr(10), ' ')}: precision={prec:.2f} "
              f"recall={recall:.2f} f1={f1:.2f}")

    x_pos = np.arange(len(method_names))  # pos = positions
    bar_w = 0.25  # w = width

    plt.figure(figsize=(9, 5.5))
    plt.bar(x_pos - bar_w, precs, width=bar_w, label="Precision",
            color="#4C72B0")
    plt.bar(x_pos, recalls, width=bar_w, label="Recall", color="#DD8452")
    plt.bar(x_pos + bar_w, f1s, width=bar_w, label="F1", color="#55A868")

    for x, values in zip(x_pos, zip(precs, recalls, f1s)):
        for offset, value in zip([-bar_w, 0, bar_w], values):
            plt.text(x + offset, value + 0.01, f"{value:.2f}",
                     ha="center", fontsize=8)

    plt.xticks(x_pos, method_names)
    plt.ylabel("Score")
    plt.ylim(0, 0.65)
    plt.title(
        f"{cur_sta['name']}: all four methods compared on the same "
        f"241 labelled days"
    )
    plt.legend()
    plt.tight_layout()

    out_path = fig_dir / "results_comparison.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"\nSaved comparison chart to {out_path}")


if __name__ == "__main__":
    main()
