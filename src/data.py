"""Download and prepare the French motor third-party liability data (freMTPL2, CASdatasets).

Cleaning follows the standard treatment in the literature (Noll, Salzmann & Wüthrich, 2018):
claim counts capped at 4 and exposures capped at 1 year.
"""
import urllib.request
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

BASE = "https://raw.githubusercontent.com/dutangc/CASdatasets/master/data/"
DRIV_BINS = [17, 20, 25, 30, 40, 50, 70, 120]
DRIV_LABELS = ["18-20", "21-25", "26-30", "31-40", "41-50", "51-70", "71+"]
VEHAGE_BINS = [-1, 0, 5, 10, 15, 200]
VEHAGE_LABELS = ["0", "1-5", "6-10", "11-15", "16+"]


def _read_rda(path, name):
    import rdata
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return rdata.conversion.convert(rdata.parser.parse_file(path))[name]


def load(data_dir):
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    out = data_dir / "fremtpl2_prepared.csv.gz"
    if out.exists():
        return pd.read_csv(out)
    for name in ("freMTPL2freq", "freMTPL2sev"):
        target = data_dir / f"{name}.rda"
        if not target.exists():
            urllib.request.urlretrieve(BASE + f"{name}.rda", target)
    freq = _read_rda(data_dir / "freMTPL2freq.rda", "freMTPL2freq")
    sev = _read_rda(data_dir / "freMTPL2sev.rda", "freMTPL2sev")
    freq = pd.DataFrame({c: np.asarray(freq[c]) for c in freq.columns})
    freq["IDpol"] = freq["IDpol"].astype(float).astype(int)
    sev = pd.DataFrame({"IDpol": np.asarray(sev["IDpol"]).astype(int),
                        "ClaimAmount": np.asarray(sev["ClaimAmount"], dtype=float)})
    amounts = sev.groupby("IDpol")["ClaimAmount"].agg(["sum", "count"]).rename(
        columns={"sum": "ClaimTotal", "count": "SevCount"})
    df = freq.merge(amounts, left_on="IDpol", right_index=True, how="left")
    df[["ClaimTotal", "SevCount"]] = df[["ClaimTotal", "SevCount"]].fillna(0)
    df["ClaimNb"] = np.minimum(df["ClaimNb"].astype(float), 4)
    df["Exposure"] = np.minimum(df["Exposure"].astype(float), 1.0)
    df.to_csv(out, index=False, compression="gzip")
    return df


def add_features(df):
    """Rating factors in the form a pricing actuary would use in a GLM tariff."""
    df = df.copy()
    df["DrivAgeBand"] = pd.cut(df["DrivAge"], DRIV_BINS, labels=DRIV_LABELS).astype(str)
    df["VehAgeBand"] = pd.cut(df["VehAge"], VEHAGE_BINS, labels=VEHAGE_LABELS).astype(str)
    df["VehPowerCat"] = np.minimum(df["VehPower"], 9).astype(int).astype(str)
    df["BonusMalusC"] = np.minimum(df["BonusMalus"], 150).astype(float)
    df["LogDensity"] = np.log(df["Density"].astype(float))
    df["AreaCode"] = df["Area"].astype(str).map({k: i for i, k in enumerate("ABCDEF", 1)}).astype(float)
    # Shape terms suggested by the GBM's SHAP dependence plots
    df["BonusMalusLog"] = np.log(df["BonusMalusC"])
    df["BonusMalusOver100"] = np.maximum(df["BonusMalusC"] - 100, 0)
    df["BonusMalusMin"] = (df["BonusMalusC"] <= 50).astype(float)
    df["LogDensitySq"] = df["LogDensity"] ** 2
    for c in ["VehBrand", "VehGas", "Region"]:
        df[c] = df[c].astype(str)
    return df
