"""
Visualisation: how the LSTM Autoencoder's F1 score changes with how long
it is trained.
"""

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras

import lstm_autoencoder_model as lstm

epoch_counts = [20, 50, 100, 150, 200]


def train_and_evaluate(windows, end_dates, labels, n_epoch):
    """Train one LSTM autoencoder for n_epoch epochs and return its
    evaluation metrics."""
    # n_epoch = number of epochs, rnd = random
    keras.utils.set_random_seed(lstm.rnd_seed)

    # m = model
    m = lstm.build_autoencoder(
        lstm.seq_len, n_features = len(lstm.feat_cols)
    )
    m.fit(windows, windows, epochs = n_epoch, batch_size = lstm.batch_sz,
          verbose = 0, shuffle = True)

    # recon/err = reconstruction/error
    recon_err = lstm.compute_reconstruction_error(
        m, windows, end_dates, len(lstm.feat_cols)
    )
    dev_score = lstm.local_deviation_score(recon_err)  # dev = deviation
    pred_noise = (dev_score > lstm.out_thr).fillna(False)  # pred = predicted

    result = pd.DataFrame({"predicted_noise": pred_noise})
    merged = labels.merge(
        result, left_on = "date", right_index = True, how = "left"
    )
    merged["predicted_noise"] = merged["predicted_noise"].fillna(False)

    return lstm.compute_metrics(merged["is_noise"], merged["predicted_noise"])


def main():
    """Retrain the LSTM autoencoder at each epoch count and plot F1
    against epochs."""
    df, filled = lstm.load_data()
    scaled, _ = lstm.scale_features(filled)
    windows, end_dates = lstm.build_sequences(scaled)
    labels = lstm.load_labels()

    print(f"Training the same model at {len(epoch_counts)} different "
          f"epoch counts: {epoch_counts}")

    rows = []
    for n_epoch in epoch_counts:
        print(f"Training for {n_epoch} epochs...")
        metrics = train_and_evaluate(windows, end_dates, labels, n_epoch)
        rows.append({"epochs": n_epoch, "precision": metrics["precision"],
                     "recall": metrics["recall"], "f1": metrics["f1"]})
        print(f"  precision={metrics['precision']:.2f} "
              f"recall={metrics['recall']:.2f} f1={metrics['f1']:.2f}")

    results = pd.DataFrame(rows)
    results.to_csv(lstm.proc_dir / "epoch_sweep_results.csv", index = False)
    print(f"\nSaved results to {lstm.proc_dir / 'epoch_sweep_results.csv'}")

    plt.figure(figsize = (8, 5))
    plt.plot(results["epochs"], results["f1"], marker = "o", label = "F1",
             linewidth = 2, color = "#55A868")
    plt.plot(results["epochs"], results["precision"], marker = "o",
             label = "Precision", linestyle = "--", alpha = 0.7)
    plt.plot(results["epochs"], results["recall"], marker = "o",
             label = "Recall", linestyle = "--", alpha = 0.7)

    best_row = results.loc[results["f1"].idxmax()]
    plt.scatter([best_row["epochs"]], [best_row["f1"]], color = "red",
                zorder = 5, s = 80,
                label = f"best F1 ({best_row['f1']:.2f} at "
                        f"{int(best_row['epochs'])} epochs)")

    plt.xlabel("Training epochs")
    plt.ylabel("Score")
    plt.title(
        f"{lstm.cur_sta['name']}: LSTM Autoencoder score vs training "
        f"length"
    )
    plt.legend()
    plt.tight_layout()

    out_path = lstm.fig_dir / "epoch_sweep.png"
    plt.savefig(out_path, dpi = 150)
    plt.close()
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    main()
