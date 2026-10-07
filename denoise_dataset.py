"""
Final stage: produce the denoised dataset using the rolling median
(Hampel) baseline filter, replacing flagged and missing SNWD readings
with the filter's own local-median estimate.
"""

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

# var/win/sz/thr/cur/sta: see baseline_filter.py
from baseline_filter import (
    rolling_median_filter, filter_var, win_sz, out_thr, cur_sta,
)
# proc = processed, fig = figures, p = path
proc_dir = Path("data/processed") / cur_sta["triplet"].replace(":", "_")
fig_dir = Path("figures") / cur_sta["triplet"].replace(":", "_")
data_p = proc_dir / "snotel_preprocessed.csv"

fallback_win = 31   # win = window


def main():
    """Fill flagged and missing SNWD readings, then save and plot the
    denoised dataset."""
    df = pd.read_csv(data_p, parse_dates = ["date"], index_col = "date")
    series = df[filter_var]

    result = rolling_median_filter(series, win_sz, out_thr)

    orig_missing = series.isna()  # orig = originally
    flag_noise = result["predicted_noise"]
    needs_repl = orig_missing | flag_noise  # repl = replacing

    denoised = series.copy()
    denoised[needs_repl] = result.loc[needs_repl, "rolling_median"]

    fill_method = pd.Series("original", index = series.index)
    fill_method[needs_repl] = "7-day window"

    still_miss_1 = denoised.isna()  # miss = missing
    if still_miss_1.any():
        # res = result
        wide_res = rolling_median_filter(series, fallback_win, out_thr)
        denoised[still_miss_1] = wide_res.loc[
            still_miss_1, "rolling_median"
        ]
        fill_method[still_miss_1 & denoised.notna()] = "31-day fallback window"

    df["SNWD_denoised"] = denoised
    df["SNWD_is_imputed"] = needs_repl
    df["SNWD_fill_method"] = fill_method

    still_miss = df["SNWD_denoised"].isna().sum()

    print(f"Total days: {len(df)}")
    print(f"Originally missing SNWD: {orig_missing.sum()}")
    print(f"Flagged as noise by the filter (kept but replaced): "
          f"{flag_noise.sum()}")
    print(f"Total points imputed: {needs_repl.sum()} "
          f"({100 * needs_repl.mean():.1f}% of the record)")
    n_7day = (fill_method == "7-day window").sum()
    n_31day = (fill_method == "31-day fallback window").sum()
    print(f"  - filled by the normal 7-day window: {n_7day}")
    print(f"  - needed the wider 31-day fallback: {n_31day}")
    if still_miss:
        print(f"Still unfillable after imputation (not enough nearby "
              f"context): {still_miss}")

    out_path = proc_dir / "snotel_denoised.csv"
    df.to_csv(out_path)
    print(f"\nSaved final denoised dataset to {out_path}")

    plt.figure(figsize = (14, 5))
    plt.plot(df.index, series, linewidth = 0.5, color = "lightsteelblue",
             label = "original raw SNWD", zorder = 1)
    plt.plot(df.index, df["SNWD_denoised"], linewidth = 0.7,
             color = "darkgreen", label = "denoised SNWD", zorder = 2)
    imp_points = df[df["SNWD_is_imputed"]]  # imp = imputed
    plt.scatter(imp_points.index, imp_points["SNWD_denoised"],
                color = "orange", s = 6,
                label = f"imputed point ({len(imp_points)})", zorder = 3)
    plt.title(
        f"{cur_sta['name']}: final denoised SNWD "
        f"(rolling median baseline filter)"
    )
    plt.xlabel("Date")
    plt.ylabel("SNWD (in)")
    plt.legend(fontsize = 8)
    plt.tight_layout()

    out_fig = fig_dir / "final_denoised_overview.png"
    plt.savefig(out_fig, dpi = 150)
    plt.close()
    print(f"Saved overview plot to {out_fig}")


if __name__ == "__main__":
    main()
