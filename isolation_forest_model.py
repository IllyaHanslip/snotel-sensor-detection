"""
ML-based denoising model #2 of 3: an Isolation Forest anomaly detector.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

from stations_config import get_primary_station

# cur/sta = current/station, proc = processed, fig = figures, p = path
cur_sta = get_primary_station()

proc_dir = Path("data/processed") / cur_sta["triplet"].replace(":", "_")
fig_dir = Path("figures") / cur_sta["triplet"].replace(":", "_")
data_p = proc_dir / "snotel_preprocessed.csv"
lab_p = proc_dir / "manual_labels.csv"

base_vars = ["SNWD", "WTEQ", "TOBS", "precip_today"]  # var = variable
# feat/cols = feature/columns
feat_cols = (
    [f"{name}_dev" for name in base_vars]
    + ["SNWD_WTEQ_mismatch", "BATT_daily_min"]
)

score_win = 7   # win = window
# contam = contamination; 'auto' uses scikit-learn's own threshold from
# the original Isolation Forest paper (Liu et al., 2008), rather than
# the labelled noise rate, since label_data.py's stratified sampling
# deliberately oversamples suspicious days and so is not a valid
# estimate of the true population contamination rate
contam = "auto"
n_ens = 15   # n_ens = number of ensemble seeds


def local_deviation_score(series, win_sz = score_win):
    """Turn a raw variable into a local-deviation suspicion score.
    min_periods = 3: see baseline_filter.py. Capped at 10.0 rather
    than infinite (unlike the other models) because this score is fed
    into IsolationForest as a fit() input, and scikit-learn's
    estimators reject infinite/NaN feature values outright."""
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
        np.where(abs_dev > 0, 10.0, 0.0),
    )
    return pd.Series(dev_score, index = series.index)


def load_data():
    """Load the data and engineer the local-deviation and mismatch
    features used by the model."""
    df = pd.read_csv(data_p, parse_dates = ["date"], index_col = "date")

    df["precip_today"] = df["PREC"].diff()
    df.loc[df["precip_today"] < 0, "precip_today"] = np.nan

    for base_name in base_vars:
        df[f"{base_name}_dev"] = local_deviation_score(df[base_name])

    df["SNWD_WTEQ_mismatch"] = df["SNWD_dev"] - df["WTEQ_dev"]

    return df


def load_labels():
    """Load the manual good/noise labels for the primary station."""
    labels = pd.read_csv(lab_p, parse_dates = ["date"])
    labels["is_noise"] = labels["label"] == "noise"
    return labels


def fit_isolation_forest(df, feat_cols = feat_cols, contam = contam,
                          n_seeds = n_ens):
    """Fit an ensemble of independently-seeded Isolation Forests and
    return their majority-vote prediction."""
    feat_tbl = df[feat_cols].dropna()  # tbl = table

    vote_n = pd.Series(0, index = feat_tbl.index)  # n = number/count
    score_sum = pd.Series(0.0, index = feat_tbl.index)
    for seed in range(n_seeds):
        m = IsolationForest(  # m = model
            contamination = contam, random_state = seed
        )
        m.fit(feat_tbl)
        vote_n += (m.predict(feat_tbl) == -1).astype(int)
        score_sum += -m.score_samples(feat_tbl)

    pred_noise = vote_n > (n_seeds / 2)  # pred = predicted
    anom_score = score_sum / n_seeds  # anom = anomaly

    return pred_noise, anom_score


def show_ensemble_size_sensitivity(
    df, lab_actual, cand_sizes = (5, 10, 15, 20, 25, 30), n_rep = 5
):
    """Check how much ensembled F1 still varies across independent
    replicates, at each candidate ensemble size."""
    # cand = candidate, n_rep = number of replicates
    print(f"\nHow much the ensembled F1 still varies across {n_rep} "
          f"independent replicates, by ensemble size:")
    print(f"{'n_seeds':>8} {'mean F1':>10} {'std F1':>10} "
          f"{'min F1':>10} {'max F1':>10}")

    rows = []
    for n_seeds in cand_sizes:
        rep_f1 = []  # rep = replicate
        for rep in range(n_rep):
            seed_off = rep * 1000  # off = offset
            feat_tbl = df[feat_cols].dropna()
            vote_n = pd.Series(0, index = feat_tbl.index)
            for seed in range(seed_off, seed_off + n_seeds):
                m = IsolationForest(
                    contamination = contam, random_state = seed
                )
                m.fit(feat_tbl)
                vote_n += (m.predict(feat_tbl) == -1).astype(int)
            pred_noise = vote_n > (n_seeds / 2)

            merged = lab_actual.merge(
                pred_noise.rename("predicted_noise"),
                left_on = "date", right_index = True, how = "inner",
            )
            metrics = compute_metrics(
                merged["is_noise"], merged["predicted_noise"]
            )
            rep_f1.append(metrics["f1"])

        rep_f1 = np.array(rep_f1)
        rows.append({
            "n_seeds": n_seeds, "mean_f1": rep_f1.mean(),
            "std_f1": rep_f1.std(),
            "min_f1": rep_f1.min(), "max_f1": rep_f1.max(),
        })
        print(f"{n_seeds:>8} {rows[-1]['mean_f1']:>10.3f} "
              f"{rows[-1]['std_f1']:>10.3f} "
              f"{rows[-1]['min_f1']:>10.3f} {rows[-1]['max_f1']:>10.3f}")

    return pd.DataFrame(rows)


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


def print_metrics_report(metrics, n_scored):  # n_scored = days scored
    """Print a confusion matrix and headline metrics to the console."""
    print("\nConfusion matrix:")
    print("                    predicted good   predicted noise")
    print(f"  actual good            {metrics['true_negative']:>6}"
          f"           {metrics['false_positive']:>6}")
    print(f"  actual noise           {metrics['false_negative']:>6}"
          f"           {metrics['true_positive']:>6}")

    print(f"\nAccuracy:  {metrics['accuracy']:.2f}  (fraction of the "
          f"{n_scored} scoreable labeled days scored correctly)")
    print(f"Precision: {metrics['precision']:.2f}  (of the days the "
          f"model called 'noise', how many really were)")
    print(f"Recall:    {metrics['recall']:.2f}  (of the days that "
          f"really were 'noise', how many the model caught)")
    print(f"F1 score:  {metrics['f1']:.2f}  (a single balance of "
          f"precision and recall)")


def show_contamination_sensitivity(df, lab_actual):
    """Print precision/recall/F1 at contamination settings 0.02-0.2
    plus 'auto', to show what the 'auto' default is trading off
    against."""
    print("\nHow the contamination setting changes precision and recall:")
    print(f"{'contamination':>14} {'precision':>10} {'recall':>10} "
          f"{'f1':>10}")

    for contam in [0.02, 0.05, 0.1, 0.15, 0.2, "auto"]:
        pred_noise, _ = fit_isolation_forest(df, contam = contam)
        merged = lab_actual.merge(
            pred_noise.rename("predicted_noise"),
            left_on = "date", right_index = True, how = "inner",
        )
        metrics = compute_metrics(
            merged["is_noise"], merged["predicted_noise"]
        )
        marker = "  <- default" if contam == contam else ""
        label = f"{contam}" if contam != "auto" else "auto"
        print(f"{label:>14} {metrics['precision']:>10.2f} "
              f"{metrics['recall']:>10.2f} {metrics['f1']:>10.2f}{marker}")


def plot_results(df, merged, out_dir = fig_dir):
    """Plot the raw series, marking caught, missed, and wrongly-flagged
    days."""
    caught_ok = merged[merged["is_noise"] & merged["predicted_noise"]]
    miss_noise = merged[merged["is_noise"] & ~merged["predicted_noise"]]
    flag_bad = merged[~merged["is_noise"] & merged["predicted_noise"]]

    plt.figure(figsize = (14, 5))
    plt.plot(df.index, df["SNWD"], linewidth = 0.6, color = "steelblue",
             label = "raw SNWD", zorder = 1)
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

    plt.title(
        f"{cur_sta['name']}: Isolation Forest (multivariate, local "
        f"deviation) vs manual labels"
    )
    plt.xlabel("Date")
    plt.ylabel("SNWD (in)")
    plt.legend(fontsize = 8)
    plt.tight_layout()

    out_path = out_dir / "isolation_forest_evaluation.png"
    plt.savefig(out_path, dpi = 150)
    plt.close()
    print(f"\nSaved evaluation plot to {out_path}")


def main():
    """Run the full Isolation Forest pipeline: fit, score, evaluate,
    and plot."""
    df = load_data()
    labels = load_labels()

    pred_noise, anom_score = fit_isolation_forest(df)
    print(f"Scored {len(pred_noise)} of {len(df)} days that had all "
          f"{len(feat_cols)} features available "
          f"(ensemble of {n_ens} independently-seeded forests, "
          f"majority vote).")

    merged = labels.merge(
        pred_noise.rename("predicted_noise"),
        left_on = "date", right_index = True, how = "inner",
    )
    merged = merged.merge(
        anom_score.rename("anomaly_score"),
        left_on = "date", right_index = True, how = "left",
    )

    n_unscored = len(labels) - len(merged)
    if n_unscored > 0:
        print(f"\nNote: {n_unscored} of {len(labels)} labeled days "
              f"couldn't be scored (missing data in their local "
              f"window) and were excluded from evaluation.")

    metrics = compute_metrics(merged["is_noise"], merged["predicted_noise"])

    n_noise = merged["is_noise"].sum()
    n_good = (~merged["is_noise"]).sum()
    print(f"\nEvaluated on {len(merged)} manually labeled days "
          f"({n_noise} noise, {n_good} good)")
    print_metrics_report(metrics, len(merged))

    out_csv = proc_dir / "isolation_forest_results.csv"
    merged.to_csv(out_csv, index = False)
    print(f"\nSaved full per-day results to {out_csv}")

    show_contamination_sensitivity(df, labels)
    ens_size_res = show_ensemble_size_sensitivity(df, labels)
    ens_csv = proc_dir / "isolation_forest_ensemble_size_results.csv"
    ens_size_res.to_csv(ens_csv, index = False)
    print(f"Saved full ensemble-size sensitivity results to {ens_csv}")
    plot_results(df, merged)


if __name__ == "__main__":
    main()
