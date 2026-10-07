"""
Interactive tool for manually labeling a sample of days as "good" or "noise".
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from stations_config import get_primary_station

# cur/sta = current/station, proc = processed, lab/p = label(s)/path
cur_sta = get_primary_station()

proc_dir = Path("data/processed") / cur_sta["triplet"].replace(":", "_")
input_path = proc_dir / "snotel_preprocessed.csv"
lab_p = proc_dir / "manual_labels.csv"

judge_var = "SNWD"   # var = variable
ctx_vars = ["SNWD", "WTEQ", "TOBS"]   # ctx = context
ctx_days = 15
rnd_seed = 42   # rnd = random
n_rand_snow = 150   # n = number
n_rand_any = 50
stuck_min_run = 5


def load_data():
    """Load the preprocessed data for the primary station."""
    df = pd.read_csv(input_path, parse_dates = ["date"], index_col = "date")
    return df


def find_stuck_value_runs(series, min_run_len = stuck_min_run):
    """Return one candidate date per run of repeated identical
    values."""
    values = series.to_numpy()
    candidates = []
    run_start = 0

    for i in range(1, len(values) + 1):
        at_end = i == len(values)
        breaks_run = (
            not at_end
            and (np.isnan(values[i]) or np.isnan(values[i - 1])
                 or values[i] != values[i - 1])
        )
        if at_end or breaks_run:
            run_len = i - run_start  # len = length
            if run_len >= min_run_len and not np.isnan(values[run_start]):
                mid_idx = run_start + run_len // 2  # mid = middle
                candidates.append(series.index[mid_idx])
            run_start = i

    return candidates


def build_candidate_list(df):
    """Build the stratified sample of candidate days to label, combining
    the five selection rules."""
    rng = np.random.default_rng(rnd_seed)
    has_val = df[judge_var].notna()  # val = value
    cand_dates = set()  # cand = candidate

    if "SNWD_raw_qc" in df.columns:
        suspect_dates = df.index[(df["SNWD_raw_qc"] == "S") & has_val]
        cand_dates.update(suspect_dates)
        print(f"Added {len(suspect_dates)} day(s) NRCS flagged Suspect "
              f"on the raw reading.")

    if "is_extreme_jump" in df.columns:
        flagged_dates = df.index[df["is_extreme_jump"] & has_val]
        cand_dates.update(flagged_dates)
        print(f"Added {len(flagged_dates)} day(s) flagged as extreme jumps.")

    stuck_dates = find_stuck_value_runs(df[judge_var])
    cand_dates.update(stuck_dates)
    print(f"Added {len(stuck_dates)} day(s) from stuck-value runs.")

    snow_days = df.index[(df[judge_var] > 0) & has_val]
    snow_samp_n = min(n_rand_snow, len(snow_days))  # samp = sample
    snow_samp_pos = rng.choice(  # pos = positions
        len(snow_days), size = snow_samp_n, replace = False
    )
    snow_samp = snow_days[snow_samp_pos]
    cand_dates.update(snow_samp)
    print(f"Added {snow_samp_n} random day(s) with snow on the ground.")

    any_days = df.index[has_val]
    any_samp_n = min(n_rand_any, len(any_days))
    any_samp_pos = rng.choice(
        len(any_days), size = any_samp_n, replace = False
    )
    any_samp = any_days[any_samp_pos]
    cand_dates.update(any_samp)
    print(f"Added {any_samp_n} random day(s) from the full record.")

    cand_dates = sorted(cand_dates)
    print(f"\nTotal candidate days to label: {len(cand_dates)}")
    return cand_dates


# lab/cols = label(s)/columns
lab_cols = [
    "date", "value", "label", "note",
    "nrcs_raw_qc_flag", "nrcs_edited_this_value",
]


def load_existing_labels():
    """Load previously saved labels, or an empty table if none exist yet."""
    if lab_p.exists():
        return pd.read_csv(lab_p, parse_dates = ["date"])
    return pd.DataFrame(columns = lab_cols)


def save_label(exist_labs, df, date, value, label, note):
    """Append one label to the saved table and write it back to disk."""
    # exist/labs = existing/labels, qc = quality control
    if "SNWD_raw_qc" in df.columns:
        nrcs_qc = df.loc[date, "SNWD_raw_qc"]
    else:
        nrcs_qc = None
    if "SNWD_official" in df.columns:
        nrcs_edited = bool(
            df.loc[date, "SNWD"] != df.loc[date, "SNWD_official"]
        )
    else:
        nrcs_edited = None

    new_row = pd.DataFrame([{
        "date": date,
        "value": value,
        "label": label,
        "note": note,
        "nrcs_raw_qc_flag": nrcs_qc,
        "nrcs_edited_this_value": nrcs_edited,
    }])
    if exist_labs.empty:
        upd_labs = new_row  # upd = updated
    else:
        upd_labs = pd.concat([exist_labs, new_row], ignore_index = True)
    upd_labs.to_csv(lab_p, index = False)
    return upd_labs


def show_context_plot(df, date, variables = ctx_vars, ctx_days = ctx_days):
    """Plot a short window of days around a candidate date, for every
    context variable."""
    win_start = date - pd.Timedelta(days = ctx_days)  # win = window
    win_end = date + pd.Timedelta(days = ctx_days)

    fig, axes = plt.subplots(
        len(variables), 1, figsize = (8, 3 * len(variables)), sharex = True
    )
    if len(variables) == 1:
        axes = [axes]

    for axis, variable in zip(axes, variables):
        window = df.loc[win_start:win_end, variable]
        axis.plot(window.index, window.values, marker = "o",
                  markersize = 3, linewidth = 1)
        axis.axvline(date, color = "red", linestyle = "--")
        axis.set_ylabel(variable)

    axes[0].set_title(f"Candidate day: {date.date()} (red dashed line)")
    axes[-1].set_xlabel("Date")
    fig.tight_layout()
    plt.show(block = False)
    plt.pause(0.1)


def run_labeling_session():
    """Interactively ask the user to label every remaining candidate
    day, saving after each answer."""
    df = load_data()
    cand_dates = build_candidate_list(df)

    labs_done = load_existing_labels()  # labs = labels
    already_done = set(pd.to_datetime(labs_done["date"]))
    remaining = [date for date in cand_dates if date not in already_done]

    print(f"\n{len(already_done)} day(s) already labeled in a "
          f"previous session.")
    print(f"{len(remaining)} day(s) left to label now.\n")
    print("For each day, type:")
    print("  g = good, n = noise, s = skip for now, q = quit and save\n")

    for date in remaining:
        value = df.loc[date, judge_var]
        show_context_plot(df, date)

        try:
            raw_ans = input(  # ans = answer
                f"{date.date()}  {judge_var} = {value}   [g/n/s/q]: "
            )
        except EOFError:
            print("\nNo more input available - stopping here, your "
                  "progress so far has been saved.")
            break

        answer = raw_ans.strip().strip("﻿").lower()
        plt.close("all")

        if answer == "q":
            print("Stopping here - your progress so far has been saved.")
            break
        elif answer == "g":
            labs_done = save_label(labs_done, df, date, value, "good", "")
        elif answer == "n":
            note = input(
                "  Optional note about why this looks like noise: "
            ).strip()
            labs_done = save_label(labs_done, df, date, value, "noise", note)
        elif answer == "s":
            continue
        else:
            print("  Not understood, skipping this one for now.")

    print(f"\n{len(labs_done)} day(s) labeled in total so far.")
    print(f"Labels saved to {lab_p}")


if __name__ == "__main__":
    run_labeling_session()
