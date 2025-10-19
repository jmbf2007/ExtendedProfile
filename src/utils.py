from __future__ import annotations
import numpy as np
import pandas as pd

def to_ticks(price: float, tick_size: float) -> int:
    return int(round(price / tick_size))

def from_ticks(ticks: int, tick_size: float) -> float:
    return round(ticks * tick_size, 10)

def bin_price_levels(df: pd.DataFrame, tick_size: float, cols=("Open","High","Low","Close")) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c+"_ticks"] = np.round(out[c] / tick_size).astype("int64")
    return out

def zscore(x: pd.Series) -> pd.Series:
    mu, sd = x.mean(), x.std(ddof=0)
    if sd == 0 or np.isnan(sd):
        return pd.Series(np.zeros(len(x)), index=x.index, dtype="float64")
    return (x - mu) / sd
