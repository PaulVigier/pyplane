"""Library for Have dream TMP."""

import numpy as np
import pandas as pd
from libs import atmos_1976 as atm


def l_over_d(df: pd.DataFrame) -> None:
    """Get L over D curve from sawtooth descents."""
    df["Altitude"] = (df["Alt_begin"] + df["Alt_end"]) / 2
    df["CAS"] = (df["CAS_begin"] + df["CAS_end"]) / 2
    df["Rho"] = atm.density(df["Altitude"])

    df["L/D"] = 1 / np.tan(np.radians(df["FPA"]))
    df["TAS"] = df.apply(lambda row: atm.CAStoTAS(row["CAS"], row["Altitude"]), axis=1)

    return df
