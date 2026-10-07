"""
ML-based denoising model #3 of 3: an LSTM Autoencoder.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers

from stations_config import get_primary_station

# cur/sta = current/station, proc = processed, fig = figures, p = path
cur_sta = get_primary_station()

proc_dir = Path("data/processed") / cur_sta["triplet"].replace(":", "_")
fig_dir = Path("figures") / cur_sta["triplet"].replace(":", "_")
data_p = proc_dir / "snotel_preprocessed.csv"
lab_p = proc_dir / "manual_labels.csv"

# feat/cols = feature/columns
feat_cols = ["SNWD", "WTEQ", "TOBS", "precip_today"]
seq_len = 14   # seq/len = sequence/length
bneck_sz = 8   # bneck/sz = bottleneck/size
n_epoch = 100
batch_sz = 32   # fixed, untuned training batch size
score_win = 7   # win = window
out_thr = 3.0   # thr = threshold
rnd_seed = 42   # rnd = random


def load_data():
    """Load the data and interpolate small gaps for the model's input
    features."""
    df = pd.read_csv(data_p, parse_dates = ["date"], index_col = "date")

    df["precip_today"] = df["PREC"].diff()
    df.loc[df["precip_today"] < 0, "precip_today"] = np.nan

    filled = df[feat_cols].interpolate(limit_direction = "both")
    return df, filled


def scale_features(filled):
    """Rescale every feature to zero mean and unit variance."""
    scaler = StandardScaler()
    scaled_vals = scaler.fit_transform(filled)  # vals = values
    scaled = pd.DataFrame(
        scaled_vals, index = filled.index, columns = filled.columns
    )
    return scaled, scaler


def build_sequences(scaled, seq_len = seq_len):
    """Slide a seq_len-day window across the data, returning every
    window and the date it ends on."""
    values = scaled.to_numpy()
    windows = []
    end_dates = []

    for start in range(len(values) - seq_len + 1):
        end = start + seq_len
        windows.append(values[start:end])
        end_dates.append(scaled.index[end - 1])

    return np.array(windows), pd.DatetimeIndex(end_dates)


def build_autoencoder(seq_len, n_features, bneck_sz = bneck_sz):
    """Build the LSTM encoder-decoder autoencoder model."""
    m = keras.Sequential([  # m = model
        layers.Input(shape = (seq_len, n_features)),
        layers.LSTM(bneck_sz, activation = "tanh",
                    return_sequences = False, name = "encoder"),
        layers.RepeatVector(seq_len),
        layers.LSTM(bneck_sz, activation = "tanh",
                    return_sequences = True, name = "decoder"),
        layers.TimeDistributed(layers.Dense(n_features)),
    ])
    m.compile(optimizer = "adam", loss = "mse")
    return m


def compute_reconstruction_error(m, windows, end_dates, n_features):
    """Reconstruct every window and return the error on each window's
    last day."""
    recon_out = m.predict(windows, verbose = 0)  # recon = reconstruction
    last_actual = windows[:, -1, :]
    last_recon = recon_out[:, -1, :]

    err = np.mean((last_actual - last_recon) ** 2, axis = 1)  # err = error
    return pd.Series(err, index = end_dates)


def local_deviation_score(series, win_sz = score_win):
    """Turn the reconstruction error into a local-deviation suspicion
    score. min_periods = 3: see baseline_filter.py."""
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
    return pd.Series(dev_score, index = series.index)


def load_labels():
    """Load the manual good/noise labels for the primary station."""
    labels = pd.read_csv(lab_p, parse_dates = ["date"])
    labels["is_noise"] = labels["label"] == "noise"
    return labels


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

    plt.title(f"{cur_sta['name']}: LSTM Autoencoder vs manual labels")
    plt.xlabel("Date")
    plt.ylabel("SNWD (in)")
    plt.legend(fontsize = 8)
    plt.tight_layout()

    out_path = out_dir / "lstm_autoencoder_evaluation.png"
    plt.savefig(out_path, dpi = 150)
    plt.close()
    print(f"\nSaved evaluation plot to {out_path}")


def main():
    """Run the full LSTM autoencoder pipeline: train, score, evaluate,
    and plot."""
    keras.utils.set_random_seed(rnd_seed)

    df, filled = load_data()
    scaled, _ = scale_features(filled)
    windows, end_dates = build_sequences(scaled)
    print(f"Built {len(windows)} overlapping {seq_len}-day windows "
          f"from {len(df)} days of data.")

    m = build_autoencoder(seq_len, n_features = len(feat_cols))
    print(m.summary())

    print(f"\nTraining for {n_epoch} epochs (this may take a minute)...")
    m.fit(windows, windows, epochs = n_epoch, batch_size = batch_sz,
          verbose = 2, shuffle = True)

    recon_err = compute_reconstruction_error(
        m, windows, end_dates, len(feat_cols)
    )
    dev_score = local_deviation_score(recon_err)
    pred_noise = (dev_score > out_thr).fillna(False)

    labels = load_labels()
    result = pd.DataFrame({
        "reconstruction_error": recon_err,
        "deviation_score": dev_score,
        "predicted_noise": pred_noise,
    })
    result.index.name = "date"

    merged = labels.merge(result, on = "date", how = "left")
    merged["predicted_noise"] = merged["predicted_noise"].fillna(False)

    n_unscored = merged["reconstruction_error"].isna().sum()
    if n_unscored > 0:
        print(f"\nNote: {n_unscored} of {len(labels)} labeled days fell "
              f"in the first {seq_len - 1} days of the record (not "
              f"enough history for a full window) and were treated as "
              f"'not flagged'.")

    metrics = compute_metrics(merged["is_noise"], merged["predicted_noise"])

    n_noise = merged["is_noise"].sum()
    n_good = (~merged["is_noise"]).sum()
    print(f"\nEvaluated on {len(merged)} manually labeled days "
          f"({n_noise} noise, {n_good} good)")
    print_metrics_report(metrics, len(merged))

    out_csv = proc_dir / "lstm_autoencoder_results.csv"
    merged.to_csv(out_csv, index = False)
    print(f"\nSaved full per-day results to {out_csv}")

    plot_results(df, merged)


if __name__ == "__main__":
    main()
