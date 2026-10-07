"""
A systematic feature ablation study for the Isolation Forest model.

Tests adding one candidate feature at a time to the current feature set,
using a paired bootstrap significance test on F1 against the current set.
"""

from pathlib import Path

import numpy as np
import pandas as pd

# cur/sta/proc/p/lab: see isolation_forest_model.py
# var = variable, feat/cols = feature/columns
from isolation_forest_model import (
    cur_sta, proc_dir, data_p, lab_p,
    base_vars, feat_cols as CURRENT_FEATURES,
    local_deviation_score, load_labels, fit_isolation_forest,
    compute_metrics,
)

cand_feats = {  # cand = candidate, feats = features
    "Soil moisture, -2in (SMS_-2_dev)": ["SMS_-2_dev"],
    "Soil moisture, -8in (SMS_-8_dev)": ["SMS_-8_dev"],
    "Soil moisture, -20in (SMS_-20_dev)": ["SMS_-20_dev"],
    "Soil temperature, -2in (STO_-2_dev)": ["STO_-2_dev"],
    "Soil temperature, -8in (STO_-8_dev)": ["STO_-8_dev"],
    "Soil temperature, -20in (STO_-20_dev)": ["STO_-20_dev"],
    "All 6 soil variables together": [
        "SMS_-2_dev", "SMS_-8_dev", "SMS_-20_dev",
        "STO_-2_dev", "STO_-8_dev", "STO_-20_dev",
    ],
    "Daily max temperature (TMAX_dev)": ["TMAX_dev"],
    "Daily min temperature (TMIN_dev)": ["TMIN_dev"],
}


def load_data_with_candidates():
    """Load the data and add local-deviation columns for every
    candidate feature."""
    df = pd.read_csv(data_p, parse_dates = ["date"], index_col = "date")

    df["precip_today"] = df["PREC"].diff()
    df.loc[df["precip_today"] < 0, "precip_today"] = np.nan

    for base_name in base_vars:
        df[f"{base_name}_dev"] = local_deviation_score(df[base_name])
    df["SNWD_WTEQ_mismatch"] = df["SNWD_dev"] - df["WTEQ_dev"]

    # cols = columns
    cand_raw_cols = [
        "SMS_-2", "SMS_-8", "SMS_-20", "STO_-2", "STO_-8", "STO_-20",
        "TMAX", "TMIN",
    ]
    for raw_col in cand_raw_cols:  # col = column
        if raw_col in df.columns:
            df[f"{raw_col}_dev"] = local_deviation_score(df[raw_col])

    return df


def evaluate_feature_set(df, labels, feat_cols):
    """Fit the ensemble Isolation Forest on one feature set and return
    its per-day predictions."""
    pred_noise, _ = fit_isolation_forest(df, feat_cols = feat_cols)
    merged = labels.merge(
        pred_noise.rename("predicted_noise"),
        left_on = "date", right_index = True, how = "inner",
    )
    return merged[["date", "is_noise", "predicted_noise"]]


def compare_on_common_days(res_a, res_b, n_boot = 2000, seed = 42):
    """Paired bootstrap test comparing two feature sets' F1 on the
    days both could score, resampling n_boot (2000) times."""
    # res = result, boot = bootstrap
    merged = res_a.merge(res_b, on = "date", suffixes = ("_a", "_b"))
    is_noise = merged["is_noise_a"].to_numpy()
    pred_a = merged["predicted_noise_a"].to_numpy()  # pred = predicted
    pred_b = merged["predicted_noise_b"].to_numpy()

    # m = metrics (here, not model)
    m_a = compute_metrics(pd.Series(is_noise), pd.Series(pred_a))
    m_b = compute_metrics(pd.Series(is_noise), pd.Series(pred_b))
    delta_obs = m_b["f1"] - m_a["f1"]  # obs = observed

    rng = np.random.default_rng(seed)
    n = len(is_noise)
    boot_deltas = np.empty(n_boot)  # boot = bootstrap
    for i in range(n_boot):
        samp_idx = rng.integers(0, n, n)  # samp = sample
        samp_is_noise = pd.Series(is_noise[samp_idx])
        samp_m_a = compute_metrics(samp_is_noise, pd.Series(pred_a[samp_idx]))
        samp_m_b = compute_metrics(samp_is_noise, pd.Series(pred_b[samp_idx]))
        boot_deltas[i] = samp_m_b["f1"] - samp_m_a["f1"]

    p_val = 2 * min((boot_deltas <= 0).mean(), (boot_deltas >= 0).mean())
    p_val = min(p_val, 1.0)  # val = value

    return n, m_a, m_b, delta_obs, p_val


def main():
    """Test every candidate feature against the current feature set
    and print a verdict for each."""
    df = load_data_with_candidates()
    labels = load_labels()

    n_feats = len(CURRENT_FEATURES)
    print(f"Baseline feature set ({n_feats} features): "
          f"{CURRENT_FEATURES}\n")
    # base/res = baseline/result
    base_res = evaluate_feature_set(df, labels, CURRENT_FEATURES)

    print(f"{'Candidate feature':<42} {'F1: base':>9} {'F1: +cand':>10} "
          f"{'delta':>7} {'p-value':>9}  Verdict")
    print("-" * 100)

    for name, extra_cols in cand_feats.items():
        available = [c for c in extra_cols if c in df.columns]
        if len(available) != len(extra_cols):
            missing = set(extra_cols) - set(available)
            print(f"{name:<42} SKIPPED - column(s) not available: "
                  f"{missing}")
            continue

        exp_feats = CURRENT_FEATURES + available  # exp = expanded
        cand_res = evaluate_feature_set(df, labels, exp_feats)

        cmp_result = compare_on_common_days(base_res, cand_res)
        n_common, base_metrics, cand_metrics, f1_delta, p_val = cmp_result

        if p_val < 0.05 and f1_delta > 0:
            verdict = "HELPS (significant)"
        elif p_val < 0.05 and f1_delta < 0:
            verdict = "HURTS (significant)"
        else:
            verdict = "no significant change"

        print(f"{name:<42} {base_metrics['f1']:>9.2f} "
              f"{cand_metrics['f1']:>10.2f} {f1_delta:>+7.2f} "
              f"{p_val:>9.3f}  {verdict}   (n={n_common})")


if __name__ == "__main__":
    main()
