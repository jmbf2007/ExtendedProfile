from __future__ import annotations
import pandas as pd
import numpy as np
from .utils import zscore

def score_levels(ep_df: pd.DataFrame, weights: dict) -> pd.DataFrame:
    if ep_df is None or ep_df.empty:
        return pd.DataFrame(columns=["Score"])
    cols = [c for c in ep_df.columns if c in weights]
    s = pd.Series(0.0, index=ep_df.index, dtype="float64")
    for c in cols:
        s = s + weights[c] * ep_df[c].astype("float64")
    out = ep_df.copy()
    out["Score"] = s
    return out.sort_values("Score", ascending=False)

def filter_levels(scored_df: pd.DataFrame, method: str, value: float, price_range=None) -> pd.DataFrame:
    if scored_df.empty:
        return scored_df
    df = scored_df.copy()
    if price_range:
        lo, hi = price_range
        df = df[(df.index >= lo) & (df.index <= hi)]
    if method == "percentile":
        thr = np.percentile(df["Score"].values, value)
        df = df[df["Score"] >= thr]
    elif method == "topN":
        df = df.sort_values("Score", ascending=False).head(int(value))
    return df

def crowd_density(levels_scores: pd.DataFrame, K: int = 3) -> pd.Series:
    if levels_scores.empty:
        return pd.Series(dtype="float64")
    idx = levels_scores.index.values
    sc = levels_scores["Score"].values
    out = np.zeros_like(sc, dtype="float64")
    for i, p in enumerate(idx):
        # simple window count weighted by 1 - |dq|/(K+1)
        w = 0.0
        for j, q in enumerate(idx):
            dq = abs(int(q) - int(p))
            if dq <= K and i != j:
                w += max(0.0, 1.0 - dq/(K+1))
        out[i] = w
    return pd.Series(out, index=levels_scores.index, name="density_k")

def crowd_aware_score(scored_df: pd.DataFrame, density_k: pd.Series, cp: pd.Series|float = 0.0, beta: float=0.5, gamma: float=0.2) -> pd.DataFrame:
    if scored_df.empty:
        return scored_df
    df = scored_df.copy()
    dens_z = (density_k - density_k.mean())/ (density_k.std(ddof=0) or 1.0)
    if isinstance(cp, (int,float)):
        cp_s = pd.Series(float(cp), index=df.index)
    else:
        cp_s = cp.reindex(df.index).fillna(0.0)
    df["Score_star"] = df["Score"] - beta*dens_z + gamma*cp_s
    return df
