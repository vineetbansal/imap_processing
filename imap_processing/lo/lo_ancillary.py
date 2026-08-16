"""Ancillary file reading for IMAP-Lo processing."""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# convert the YYYYDDD datetime format directly upon reading
_CONVERTERS = {
    "YYYYDDD": lambda x: pd.to_datetime(str(x), format="%Y%j"),
    "#YYYYDDD": lambda x: pd.to_datetime(str(x), format="%Y%j"),
    "YYYYDDD_strt": lambda x: pd.to_datetime(str(x), format="%Y%j"),
    "YYYYDDD_end": lambda x: pd.to_datetime(str(x), format="%Y%j"),
}

# Columns in the csv files to rename for consistency
_RENAME_COLUMNS = {
    "YYYYDDD": "Date",
    "#YYYYDDD": "Date",
    "#Comments": "Comments",
    "YYYYDDD_strt": "StartDate",
    "YYYYDDD_end": "EndDate",
}


def read_ancillary_file(ancillary_file: str | Path) -> pd.DataFrame:
    """
    Read a generic ancillary CSV file into a pandas DataFrame.

    Parameters
    ----------
    ancillary_file : str or Path
        Path to the ancillary CSV file.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the ancillary data.
    """
    legacy_format = False
    read_csv_kwargs: dict[str, Any] = {}
    if "esa-mode-lut" in str(ancillary_file):
        # skip the first row which is a comment
        read_csv_kwargs["skiprows"] = [0]
    elif "geometric-factor" in str(ancillary_file):
        # legacy format - rows with comment headers indicating Hi_Res and Hi_Thr
        legacy_format = "Hi_Thr,,," in Path(ancillary_file).read_text()
        if legacy_format:
            read_csv_kwargs["skiprows"] = [1, 38]
        else:
            read_csv_kwargs["comment"] = "#"
    df = pd.read_csv(ancillary_file, converters=_CONVERTERS, **read_csv_kwargs)
    df = df.rename(columns=_RENAME_COLUMNS)

    if "geometric-factor" in str(ancillary_file):
        if legacy_format and "esa_mode" not in df.columns:
            # Add an ESA mode column based on the known structure of the file.
            # The first 36 rows are ESA mode 0 (HiRes), the second 36 are ESA mode 1
            # (HiThr)
            df["esa_mode"] = 0
            df.loc[36:, "esa_mode"] = 1

    return df


def get_nominal_pivot_angle(
    anc_dependencies: list, pointing_date: np.datetime64
) -> float | None:
    """
    Look up the nominal pivot angle scheduled for a pointing.

    The "pointing-file" ancillary gives the nominal pivot angle for each day of
    the mission. It is the best estimate available for a pointing whose actual
    pivot angle cannot be measured from housekeeping.

    Parameters
    ----------
    anc_dependencies : list
        Ancillary file paths, which may or may not include the pointing file.
    pointing_date : numpy.datetime64
        The date the pointing starts on.

    Returns
    -------
    nominal_pivot_angle : float | None
        The scheduled pivot angle in degrees, or None when the pointing file is
        not among the dependencies or has no entry for the date.
    """
    pointing_file = next(
        (str(path) for path in anc_dependencies if "pointing-file" in str(path)), None
    )
    if pointing_file is None:
        return None

    pointing_angles = read_ancillary_file(pointing_file)
    scheduled = pointing_angles[
        pointing_angles["Date"] == pd.Timestamp(pointing_date).normalize()
    ]
    if scheduled.empty:
        return None

    return float(scheduled["Nominal_Angle (deg)"].iloc[0])
