from __future__ import annotations
import pandas as pd
import numpy as np

def infer_sessions(df: pd.DataFrame, gap_minutes: int = 30) -> pd.DataFrame:
    """Infer NewSession as True when the time gap between consecutive bars > gap_minutes.

    If df already has 'NewSession', we keep it but ensure first row of any detected gap also marks True.
    """
    out = df.copy()
    if "NewSession" not in out.columns:
        out["NewSession"] = False
    t = out["Time"].view("int64")//1_000_000_000
    dt = np.diff(t, prepend=t[0])
    gaps = dt > gap_minutes*60
    out.loc[gaps, "NewSession"] = True
    out.loc[out.index.min(), "NewSession"] = True
    out["SessionId"] = out["NewSession"].cumsum()
    return out
