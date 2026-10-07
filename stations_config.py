"""
Single source of truth for which SNOTEL stations this project uses.
"""

from pathlib import Path

import pandas as pd

csv_p_default = Path(__file__).parent / "stations.csv"  # p = path


def load_stations(csv_p = csv_p_default):
    """Return the full station table from stations.csv."""
    return pd.read_csv(csv_p)


def get_primary_station(csv_p = csv_p_default):
    """Return the one station with manual labels, as a plain dict."""
    stations = load_stations(csv_p)
    pri_rows = stations[stations["role"] == "primary"]  # pri = primary
    if len(pri_rows) != 1:
        raise ValueError(
            f"Expected exactly one station with role='primary' in {csv_p}, "
            f"found {len(pri_rows)}. There must be exactly one - it's the "
            f"station manual_labels.csv was built from."
        )
    return pri_rows.iloc[0].to_dict()


def station_folder_name(triplet):
    """Convert a station triplet into a filesystem-safe folder name."""
    return triplet.replace(":", "_")
