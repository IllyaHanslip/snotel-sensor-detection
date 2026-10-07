"""
Data acquisition and pre-processing for a real-world snow sensor time series.

Source: NRCS SNOTEL network, accessed via the public AWDB REST API.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt

from stations_config import load_stations, station_folder_name

# url = URL
awdb_url = "https://wcc.sc.egov.usda.gov/awdbRestApi/services/v1/data"

elements = ["SNWD", "WTEQ", "TOBS", "PREC", "TMAX", "TMIN"]

# elems = elements
depth_elems = [
    "SMS:-2", "SMS:-8", "SMS:-20", "STO:-2", "STO:-8", "STO:-20",
]

start_dt = "2015-10-01"  # dt = date
end_dt = "2023-09-30"
duration = "DAILY"


def get_station_dirs(sta_trip):  # sta = station, trip = triplet
    """Create (if needed) and return station's raw/processed/figures
    folders."""
    safe_name = station_folder_name(sta_trip)
    raw_dir = Path("data/raw") / safe_name
    proc_dir = Path("data/processed") / safe_name  # proc = processed
    fig_dir = Path("figures") / safe_name  # fig = figures

    for folder in (raw_dir, proc_dir, fig_dir):
        folder.mkdir(parents = True, exist_ok = True)

    return raw_dir, proc_dir, fig_dir


def fetch_snotel_data(sta_trip, elements = elements, start_dt = start_dt,
                       end_dt = end_dt, duration = duration):
    """Download station's daily data from the SNOTEL API."""
    req_params = {  # req = request
        "stationTriplets": sta_trip,
        "elements": ",".join(elements),
        "duration": duration,
        "beginDate": start_dt,
        "endDate": end_dt,
        "returnFlags": "true",
        "returnOriginalValues": "true",
    }

    print(f"Requesting data for {sta_trip} from the SNOTEL API...")
    response = requests.get(awdb_url, params = req_params, timeout = 30)
    response.raise_for_status()
    return response.json()


def fetch_battery_voltage(sta_trip, start_dt = start_dt, end_dt = end_dt):
    """Download hourly battery voltage and reduce it to one daily
    minimum value per day."""
    req_params = {
        "stationTriplets": sta_trip,
        "elements": "BATT",
        "duration": "HOURLY",
        "beginDate": start_dt,
        "endDate": end_dt,
        "returnFlags": "true",
        "returnOriginalValues": "true",
    }

    print(f"Requesting hourly battery voltage for {sta_trip}...")
    response = requests.get(awdb_url, params = req_params, timeout = 60)
    response.raise_for_status()
    payload = response.json()

    hourly_tbl = json_to_dataframe(payload)  # tbl = table
    hourly_tbl.index = hourly_tbl.index.normalize()
    daily_min_v = hourly_tbl["BATT"].groupby(level = 0).min()  # v = voltage
    daily_min_v.name = "BATT_daily_min"
    return daily_min_v


def fetch_depth_qualified_elements(sta_trip, elements = depth_elems,
                                    start_dt = start_dt, end_dt = end_dt):
    """Download soil moisture/temperature at three depths and return
    them as one table, keyed by column name per depth."""
    req_params = {
        "stationTriplets": sta_trip,
        "elements": ",".join(elements),
        "duration": "DAILY",
        "beginDate": start_dt,
        "endDate": end_dt,
        "returnFlags": "true",
        "returnOriginalValues": "true",
    }

    print(f"Requesting soil moisture/temperature for {sta_trip}...")
    response = requests.get(awdb_url, params = req_params, timeout = 60)
    response.raise_for_status()
    payload = response.json()

    var_recs = []  # var = variable, recs = records
    for record in payload:
        var_recs.extend(record.get("data", [record]))

    var_tbls = []  # tbls = tables
    for record in var_recs:
        elem_info = record.get("stationElement", {})  # elem = element
        code = record.get("elementCode") or elem_info.get("elementCode")
        depth = elem_info.get("heightDepth", "")
        col_name = f"{code}_{depth}" if depth != "" else code  # col = column

        daily_vals = record.get("values", [])  # vals = values
        if len(daily_vals) == 0:
            continue

        table = pd.DataFrame(daily_vals)
        table["date"] = pd.to_datetime(table["date"])
        # val = value
        if "origValue" in table.columns:
            raw_val = table["origValue"].fillna(table["value"])
        else:
            raw_val = table["value"]
        var_tbls.append(pd.DataFrame(
            {col_name: raw_val.values}, index = table["date"]
        ))

    if not var_tbls:
        print("Warning: no soil moisture/temperature data returned "
              "for this station.")
        return pd.DataFrame()

    combined = pd.concat(var_tbls, axis = 1).sort_index()
    combined.index.name = "date"
    return combined


def json_to_dataframe(payload):
    """Convert the raw API response into a table with a raw and
    official column per variable."""
    var_recs = []
    for record in payload:
        if "data" in record:
            for sub_rec in record["data"]:  # sub_rec = sub-record
                var_recs.append(sub_rec)
        else:
            var_recs.append(record)

    var_tbls = []
    for record in var_recs:
        if "elementCode" in record:
            var_name = record["elementCode"]  # name = variable name
        else:
            var_name = record["stationElement"]["elementCode"]

        daily_vals = record.get("values", [])
        if len(daily_vals) == 0:
            continue

        table = pd.DataFrame(daily_vals)
        table["date"] = pd.to_datetime(table["date"])

        if "origValue" in table.columns:
            raw_val = table["origValue"].fillna(table["value"])
        else:
            raw_val = table["value"]

        if "origQcFlag" in table.columns:
            raw_qc = table["origQcFlag"]  # qc = quality control
        else:
            raw_qc = table.get("qcFlag")

        var_tbl = pd.DataFrame({
            var_name: raw_val,
            f"{var_name}_raw_qc": raw_qc,
            f"{var_name}_official": table["value"],
            f"{var_name}_official_qc": table.get("qcFlag"),
        })
        var_tbl.index = table["date"]
        var_tbls.append(var_tbl)

    if len(var_tbls) == 0:
        raise ValueError(
            "No data was found in the API response. This usually means the "
            "station triplet does not exist, or has no data for this date "
            "range. Double check the station ID using the /stations endpoint."
        )

    combined_tbl = pd.concat(var_tbls, axis = 1)  # combined = combined table
    combined_tbl.index.name = "date"
    combined_tbl = combined_tbl.sort_index()
    return combined_tbl


def preprocess(df, elements = elements):
    """Fill calendar gaps and null out error codes (-99.9, -9999,
    9999), impossible negatives, and implausible SNWD spikes; flag
    large jumps."""
    df = df.copy()

    full_calendar = pd.date_range(df.index.min(), df.index.max(), freq = "D")
    days_added = full_calendar.difference(df.index).size
    df = df.reindex(full_calendar)
    df.index.name = "date"
    print(f"Added {days_added} missing day(s) so the calendar has no gaps.")

    err_codes = [-99.9, -9999, 9999]  # err = error
    df[elements] = df[elements].replace(err_codes, np.nan)

    for col_name in ["SNWD", "WTEQ", "PREC"]:  # col = column
        if col_name in df.columns:
            is_neg = df[col_name] < 0  # neg = negative
            n_neg = is_neg.sum()  # n = number/count
            df.loc[is_neg, col_name] = np.nan
            if n_neg > 0:
                print(f"Removed {n_neg} impossible negative {col_name} "
                      f"reading(s).")

    if "SNWD" in df.columns and "SNWD_official" in df.columns:
        ceil_plaus = df["SNWD_official"].max() * 1.25  # plaus = plausible
        impossible = df["SNWD"] > ceil_plaus
        n_impossible = impossible.sum()
        df.loc[impossible, "SNWD"] = np.nan
        if n_impossible > 0:
            print(f"Removed {n_impossible} SNWD reading(s) above "
                  f"{ceil_plaus:.0f} inches (1.25x this station's "
                  f"highest-ever validated depth) - almost certainly "
                  f"sensor faults.")

    if "SNWD" in df.columns:
        # 99.9th percentile: deliberately conservative, since this is
        # only one of five candidate-selection rules feeding manual
        # review (Section 3.3), not a final noise decision on its own
        day_chg = df["SNWD"].diff().abs()  # chg = change
        jump_thr = day_chg.quantile(0.999)  # thr = threshold
        df["is_extreme_jump"] = day_chg > jump_thr
        n_flagged = df["is_extreme_jump"].sum()
        print(f"Flagged {n_flagged} unusually large jump(s) in snow depth "
              f"(bigger than {jump_thr:.2f} inches in one day).")

    df["is_missing"] = df[elements].isna().any(axis = 1)

    return df


def summarise_and_plot(df, sta_trip, fig_dir, elements = elements):
    """Print summary statistics and save a raw-vs-official overview
    plot for every variable."""
    print("\nBasic statistics for each variable (raw sensor readings):")
    print(df[elements].describe())

    print("\nPercentage of missing values for each variable:")
    print(df[elements].isna().mean() * 100)

    print("\nHow much NRCS's official record differs from the raw "
          "sensor reading:")
    for col_name in elements:
        # official = NRCS-edited value
        col_official = f"{col_name}_official"
        if col_official not in df.columns:
            continue
        both_present = df[col_name].notna() & df[col_official].notna()
        differs = (
            df.loc[both_present, col_name]
            != df.loc[both_present, col_official]
        )
        print(f"  {col_name}: {differs.sum()} of {both_present.sum()} "
              f"day(s) edited by NRCS ({100 * differs.mean():.1f}%)")

    fig, axes = plt.subplots(
        len(elements), 1, figsize = (12, 3 * len(elements)), sharex = True
    )
    if len(elements) == 1:
        axes = [axes]

    for axis, col_name in zip(axes, elements):
        axis.plot(df.index, df[col_name], linewidth = 0.8,
                  label = "raw sensor reading")

        col_official = f"{col_name}_official"
        if col_official in df.columns:
            axis.plot(df.index, df[col_official], linewidth = 0.8,
                      color = "gray", alpha = 0.6,
                      label = "NRCS official (edited) value")

        if col_name == "SNWD" and "is_extreme_jump" in df.columns:
            flagged_rows = df[df["is_extreme_jump"]]
            axis.scatter(flagged_rows.index, flagged_rows[col_name],
                         color = "red", s = 10, label = "flagged jump")

        axis.set_ylabel(col_name)
        axis.legend(fontsize = 8)

    axes[-1].set_xlabel("Date")
    fig.suptitle(
        f"SNOTEL station {sta_trip}: raw sensor data vs NRCS's "
        f"official record"
    )
    fig.tight_layout()

    out_p = fig_dir / "raw_series_overview.png"  # p = path
    fig.savefig(out_p, dpi = 150)
    plt.close(fig)
    print(f"\nSaved plot to {out_p}")


def process_one_station(sta_trip, station_name):
    """Run the full fetch, clean, join, and save pipeline for one
    station."""
    print("=" * 70)
    print(f"Processing {station_name} ({sta_trip})")
    print("=" * 70)

    raw_dir, proc_dir, fig_dir = get_station_dirs(sta_trip)

    raw_json = fetch_snotel_data(sta_trip)
    raw_tbl = json_to_dataframe(raw_json)  # tbl = table
    raw_tbl.to_csv(raw_dir / "snotel_raw.csv")
    print(f"Saved raw data: {raw_tbl.shape[0]} rows, "
          f"{raw_tbl.shape[1]} columns")

    clean_tbl = preprocess(raw_tbl)

    # batt/v = battery/voltage
    batt_v = fetch_battery_voltage(sta_trip)
    clean_tbl = clean_tbl.join(batt_v)
    n_miss_batt = clean_tbl["BATT_daily_min"].isna().sum()  # miss = missing
    print(f"Joined battery voltage ({n_miss_batt} day(s) with no "
          f"battery reading available).")

    soil_data = fetch_depth_qualified_elements(sta_trip)
    if not soil_data.empty:
        clean_tbl = clean_tbl.join(soil_data)
        print(f"Joined {soil_data.shape[1]} soil moisture/temperature "
              f"column(s) (candidate features only).")

    clean_tbl.to_csv(proc_dir / "snotel_preprocessed.csv")
    print(f"Saved cleaned data: {clean_tbl.shape[0]} rows, "
          f"{clean_tbl.shape[1]} columns")

    summarise_and_plot(clean_tbl, sta_trip, fig_dir)
    print()


def main():
    """Process every station listed in stations.csv."""
    stations = load_stations()
    print(f"Processing {len(stations)} station(s) from stations.csv\n")
    for _, station in stations.iterrows():
        process_one_station(station["triplet"], station["name"])


if __name__ == "__main__":
    main()
