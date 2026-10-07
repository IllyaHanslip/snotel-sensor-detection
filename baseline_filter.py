"""
Baseline filter: rolling median (Hampel) filter for flagging noisy SNWD
readings, evaluated against the manually labeled "good"/"noise" set.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from stations_config import get_primary_station

# cur/sta = current/station, proc = processed, fig = figures, p = path
cur_sta = get_primary_station()

proc_dir = Path("data/processed") / cur_sta["triplet"].replace(":", "_")
fig_dir = Path("figures") / cur_sta["triplet"].replace(":", "_")
data_p = proc_dir / "snotel_preprocessed.csv"
lab_p = proc_dir / "manual_labels.csv"

filter_var = "SNWD"  # var = variable (the one being filtered)
win_sz = 7   # win/sz = window/size
out_thr = 3.0   # thr = threshold


def load_full_series():
    """Load the preprocessed SNWD series for the primary station."""
    df = pd.read_csv(data_p, parse_dates = ["date"], index_col = "date")
    return df[filter_var]


def load_labels():
    """Load the manual good/noise labels for the primary station."""
    return pd.read_csv(lab_p, parse_dates = ["date"])


def rolling_median_filter(series, win_sz = win_sz, thr = out_thr):
    """Flag days whose value deviates too far from its local rolling
    median. min_periods = 3 is the smallest window size a median and
    MAD both carry any real spread information from, rather than just
    reflecting the two endpoints."""
    # roll_med = rolling median, abs_dev = absolute deviation
    roll_med = series.rolling(
        window = win_sz, center = True, min_periods = 3
    ).median()
    abs_dev = (series - roll_med).abs()

    mad = abs_dev.rolling(
        window = win_sz, center = True, min_periods = 3
    ).median()
    mad_scaled = mad * 1.4826

    dev_score = np.where(  # dev = deviation
        mad_scaled > 0,
        abs_dev / mad_scaled,
        np.where(abs_dev > 0, np.inf, 0.0),
    )
    dev_score = pd.Series(dev_score, index = series.index)
    pred_noise = (dev_score > thr).fillna(False)  # pred = predicted

    result = pd.DataFrame({
        "rolling_median": roll_med,
        "deviation_score": dev_score,
        "predicted_noise": pred_noise,
    })
    result.index.name = "date"
    return result


def compute_metrics(actual_is_noise, pred_is_noise):
    """Compute accuracy, precision, recall, and F1 from predicted vs
    actual noise flags."""
    # tp/fp/fn/tn = true/false positive/negative
    tp = (actual_is_noise & pred_is_noise).sum()
    fp = (~actual_is_noise & pred_is_noise).sum()
    fn = (actual_is_noise & ~pred_is_noise).sum()
    tn = (~actual_is_noise & ~pred_is_noise).sum()

    total = tp + fp + fn + tn
    acc = (tp + tn) / total  # acc = accuracy

    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0  # prec = precision
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * recall / (prec + recall) if (prec + recall) > 0 else 0.0

    return {
        "true_positive": tp, "false_positive": fp,
        "false_negative": fn, "true_negative": tn,
        "accuracy": acc, "precision": prec, "recall": recall, "f1": f1,
    }


def print_metrics_report(metrics, n_lab):  # n_lab = number of labeled days
    """Print a confusion matrix and headline metrics to the console."""
    print("Confusion matrix:")
    print("                    predicted good   predicted noise")
    print(f"  actual good            {metrics['true_negative']:>6}"
          f"           {metrics['false_positive']:>6}")
    print(f"  actual noise           {metrics['false_negative']:>6}"
          f"           {metrics['true_positive']:>6}")

    print(f"\nAccuracy:  {metrics['accuracy']:.2f}  (fraction of all "
          f"{n_lab} labeled days scored correctly)")
    print(f"Precision: {metrics['precision']:.2f}  (of the days the "
          f"filter called 'noise', how many really were)")
    print(f"Recall:    {metrics['recall']:.2f}  (of the days that "
          f"really were 'noise', how many the filter caught)")
    print(f"F1 score:  {metrics['f1']:.2f}  (a single balance of "
          f"precision and recall)")


def show_threshold_sensitivity(series, lab_actual):
    """Print precision/recall/F1 at threshold values 1.5-5.0, to show
    what the default of 3.0 is trading off against."""
    print("\nHow the threshold choice trades precision off against recall:")
    print(f"{'threshold':>10} {'precision':>10} {'recall':>10} {'f1':>10}")

    for thr in [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0]:
        result = rolling_median_filter(series, win_sz, thr)
        merged = lab_actual.merge(result, on = "date", how = "left")
        metrics = compute_metrics(
            merged["is_noise"], merged["predicted_noise"]
        )
        marker = "  <- default" if thr == out_thr else ""
        print(f"{thr:>10.1f} {metrics['precision']:>10.2f} "
              f"{metrics['recall']:>10.2f} {metrics['f1']:>10.2f}{marker}")


def plot_results(series, result, lab_actual, out_dir = fig_dir):
    """Plot the raw series, marking caught, missed, and wrongly-flagged
    days."""
    merged = lab_actual.merge(result, on = "date", how = "left")

    caught_ok = merged[merged["is_noise"] & merged["predicted_noise"]]
    miss_noise = merged[merged["is_noise"] & ~merged["predicted_noise"]]
    flag_bad = merged[~merged["is_noise"] & merged["predicted_noise"]]

    plt.figure(figsize = (14, 5))
    plt.plot(series.index, series.values, linewidth = 0.6,
             color = "steelblue", label = "raw SNWD", zorder = 1)
    plt.scatter(caught_ok["date"], caught_ok["value"], color = "green",
                s = 25,
                label = f"noise, correctly caught ({len(caught_ok)})",
                zorder = 3)
    plt.scatter(miss_noise["date"], miss_noise["value"], color = "orange",
                s = 25, label = f"noise, missed ({len(miss_noise)})",
                zorder = 3)
    plt.scatter(flag_bad["date"], flag_bad["value"], color = "red", s = 25,
                label = f"good day, wrongly flagged ({len(flag_bad)})",
                zorder = 3)

    plt.title(f"{cur_sta['name']}: rolling median filter vs manual labels")
    plt.xlabel("Date")
    plt.ylabel("SNWD (in)")
    plt.legend(fontsize = 8)
    plt.tight_layout()

    out_path = out_dir / "baseline_filter_evaluation.png"
    plt.savefig(out_path, dpi = 150)
    plt.close()
    print(f"\nSaved evaluation plot to {out_path}")


def main():
    """Run the full baseline pipeline: filter, evaluate, and plot."""
    series = load_full_series()
    labels = load_labels()
    labels["is_noise"] = labels["label"] == "noise"

    result = rolling_median_filter(series)

    merged = labels.merge(result, on = "date", how = "left")
    metrics = compute_metrics(merged["is_noise"], merged["predicted_noise"])

    n_noise = labels["is_noise"].sum()
    n_good = (~labels["is_noise"]).sum()
    print(f"Evaluated on {len(labels)} manually labeled days "
          f"({n_noise} noise, {n_good} good)\n")
    print_metrics_report(metrics, len(labels))

    out_csv = proc_dir / "baseline_filter_results.csv"
    merged.to_csv(out_csv, index = False)
    print(f"\nSaved full per-day results to {out_csv}")

    show_threshold_sensitivity(series, labels)
    plot_results(series, result, labels)


if __name__ == "__main__":
    main()
