"""
Visualisation: does pooling more stations' data actually help?
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from tensorflow import keras

import isolation_forest_model as iso
import lstm_autoencoder_model as lstm
from stations_config import load_stations, station_folder_name

pool_sizes = [1, 2, "all"]


def get_pool_station_triplets():
    """Return the primary station's triplet and the list of pool-only
    station triplets."""
    stations = load_stations()
    pri_trip = lstm.cur_sta["triplet"]  # pri = primary, trip = triplet
    pool_trips = stations.loc[stations["role"] == "pool", "triplet"].tolist()
    return pri_trip, pool_trips


def stations_for_size(pri_trip, pool_trips, size):
    """Return the list of station triplets to pool for one candidate
    pool size."""
    if size == 1:
        return [pri_trip]
    if size == "all":
        return [pri_trip] + pool_trips
    return [pri_trip] + pool_trips[: size - 1]


def build_isolation_forest_features(triplet):
    """Load one station's data and engineer its Isolation Forest
    feature columns."""
    path = (Path("data/processed") / station_folder_name(triplet)
            / "snotel_preprocessed.csv")
    df = pd.read_csv(path, parse_dates = ["date"], index_col = "date")
    df["precip_today"] = df["PREC"].diff()
    df.loc[df["precip_today"] < 0, "precip_today"] = np.nan
    for base_name in iso.base_vars:
        df[f"{base_name}_dev"] = iso.local_deviation_score(df[base_name])
    df["SNWD_WTEQ_mismatch"] = df["SNWD_dev"] - df["WTEQ_dev"]
    return df[iso.feat_cols].dropna()


def run_isolation_forest_pooled(triplets):
    """Train the ensembled Isolation Forest on pooled stations' data
    and evaluate on the primary station's labels."""
    pri_tbl = build_isolation_forest_features(triplets[0])  # tbl = table
    other_tbls = [build_isolation_forest_features(t) for t in triplets[1:]]
    pooled_tbl = pd.concat([pri_tbl] + other_tbls, ignore_index = True)
    pri_row_n = len(pri_tbl)  # n = number/count

    vote_n = pd.Series(0, index = pri_tbl.index)
    for seed in range(iso.n_ens):
        m = IsolationForest(  # m = model
            contamination = iso.contam, random_state = seed
        )
        m.fit(pooled_tbl)
        is_anom = m.predict(pooled_tbl) == -1  # anom = anomaly
        vote_n += pd.Series(
            is_anom[:pri_row_n], index = pri_tbl.index
        ).astype(int)

    pred_noise = vote_n > (iso.n_ens / 2)

    labels = iso.load_labels()
    result = pd.DataFrame({"predicted_noise": pred_noise})
    merged = labels.merge(
        result, left_on = "date", right_index = True, how = "inner"
    )
    return iso.compute_metrics(merged["is_noise"], merged["predicted_noise"])


def build_lstm_filled(triplet):
    """Load one station's data and interpolate it for the LSTM's input
    features."""
    path = (Path("data/processed") / station_folder_name(triplet)
            / "snotel_preprocessed.csv")
    df = pd.read_csv(path, parse_dates = ["date"], index_col = "date")
    df["precip_today"] = df["PREC"].diff()
    df.loc[df["precip_today"] < 0, "precip_today"] = np.nan
    return df[lstm.feat_cols].interpolate(limit_direction = "both")


def run_lstm_pooled(triplets):
    """Train the LSTM autoencoder on pooled stations' data and
    evaluate on the primary station's labels."""
    # sta = station
    filled_by_sta = {t: build_lstm_filled(t) for t in triplets}

    scaler = StandardScaler()
    scaler.fit(pd.concat(filled_by_sta.values()))

    all_windows = []
    pri_windows, pri_end_dates = None, None  # pri = primary
    for t in triplets:
        scaled = pd.DataFrame(
            scaler.transform(filled_by_sta[t]),
            index = filled_by_sta[t].index,
            columns = filled_by_sta[t].columns,
        )
        windows, end_dates = lstm.build_sequences(scaled)
        all_windows.append(windows)
        if t == triplets[0]:
            pri_windows, pri_end_dates = windows, end_dates

    pooled_windows = np.concatenate(all_windows, axis = 0)

    keras.utils.set_random_seed(lstm.rnd_seed)
    m = lstm.build_autoencoder(
        lstm.seq_len, n_features = len(lstm.feat_cols)
    )
    m.fit(pooled_windows, pooled_windows, epochs = lstm.n_epoch,
          batch_size = lstm.batch_sz, verbose = 0, shuffle = True)

    recon_err = lstm.compute_reconstruction_error(
        m, pri_windows, pri_end_dates, len(lstm.feat_cols)
    )
    dev_score = lstm.local_deviation_score(recon_err)
    pred_noise = (dev_score > lstm.out_thr).fillna(False)

    labels = lstm.load_labels()
    result = pd.DataFrame({"predicted_noise": pred_noise})
    merged = labels.merge(
        result, left_on = "date", right_index = True, how = "left"
    )
    merged["predicted_noise"] = merged["predicted_noise"].fillna(False)
    return lstm.compute_metrics(merged["is_noise"], merged["predicted_noise"])


def main():
    """Run both models at each pool size and plot the effect on F1."""
    pri_trip, pool_trips = get_pool_station_triplets()
    print(f"Primary (evaluation) station: {pri_trip}")
    print(f"Available pool stations: {pool_trips}\n")

    rows = []
    for size in pool_sizes:
        triplets = stations_for_size(pri_trip, pool_trips, size)
        sta_count = len(triplets)  # sta = station
        print(f"--- Pooling {sta_count} station(s): {triplets} ---")

        iso_metrics = run_isolation_forest_pooled(triplets)
        print(f"Isolation Forest: precision={iso_metrics['precision']:.2f} "
              f"recall={iso_metrics['recall']:.2f} f1={iso_metrics['f1']:.2f}")

        lstm_metrics = run_lstm_pooled(triplets)
        print(f"LSTM Autoencoder: precision={lstm_metrics['precision']:.2f} "
              f"recall={lstm_metrics['recall']:.2f} "
              f"f1={lstm_metrics['f1']:.2f}\n")

        rows.append({
            "station_count": sta_count,
            "isolation_forest_f1": iso_metrics["f1"],
            "lstm_f1": lstm_metrics["f1"],
        })

    results = pd.DataFrame(rows)
    results.to_csv(iso.proc_dir / "station_pooling_results.csv", index = False)
    print(f"Saved results to {iso.proc_dir / 'station_pooling_results.csv'}")

    plt.figure(figsize = (8, 5))
    plt.plot(results["station_count"], results["isolation_forest_f1"],
             marker = "o", linewidth = 2, label = "Isolation Forest")
    plt.plot(results["station_count"], results["lstm_f1"], marker = "o",
             linewidth = 2, label = "LSTM Autoencoder")
    plt.xticks(results["station_count"])
    plt.xlabel("Number of stations pooled for training")
    plt.ylabel("F1 (evaluated on the primary station's labels only)")
    plt.title(
        f"{lstm.cur_sta['name']}: effect of pooling more stations' "
        f"training data"
    )
    plt.legend()
    plt.tight_layout()

    out_path = iso.fig_dir / "station_pooling_effect.png"
    plt.savefig(out_path, dpi = 150)
    plt.close()
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    main()
