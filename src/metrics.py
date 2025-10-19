from __future__ import annotations
import pandas as pd
import numpy as np

def hue(hits: int, exposure: int) -> float:
    return hits / max(1, exposure)

def hazard_first_touch(hits: int, exposure: int) -> float:
    return 1.0 - np.exp(- hits / max(1, exposure))

def contrast_lift(hue_series: pd.Series, p: int, M: int = 8, method: str = "median") -> float:
    if p not in hue_series.index:
        return np.nan
    idx = hue_series.index
    neigh = hue_series.loc[(idx >= p-M) & (idx <= p+M) & (idx != p)]
    if neigh.empty:
        return np.nan
    ref = neigh.median() if method == "median" else neigh.mean()
    return float(hue_series.loc[p] - ref)

def isolation_from_density(density_k: float) -> float:
    return float(np.exp(-density_k))

def reactivity_index(HUE: float, lam1: float, CP: float, CL: float, I: float, a=0.5, b=0.25, c=1.0, d=0.5) -> float:
    import math
    raw = math.log1p(HUE) + a*math.log1p(lam1) + b*CP + c*CL + d*math.log(max(I, 1e-9))
    return raw
