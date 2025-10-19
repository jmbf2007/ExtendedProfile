from __future__ import annotations
import pandas as pd
import numpy as np
from .utils import to_ticks

EP_COLS = ["High","Low","Close","Open","MVC","UpperWick","LowerWick"]

def _count_at_levels(df_slice: pd.DataFrame, tick_size: float) -> pd.DataFrame:
    counts = {}
    # Basic counts by tick for each column if present
    for col in EP_COLS:
        if col in df_slice.columns:
            ticks = np.round(df_slice[col].values / tick_size).astype("int64")
            vc = pd.Series(ticks).value_counts()
            counts[col] = vc
    # Build unified index
    all_levels = sorted(set().union(*[s.index for s in counts.values()])) if counts else []
    out = pd.DataFrame(index=all_levels)
    for col, vc in counts.items():
        out[col] = vc
    out = out.fillna(0).astype("int64")
    out.index.name = "level_ticks"
    return out

def build_ep(df: pd.DataFrame, session_id: int, n_prev_sessions: int, tick_size: float) -> pd.DataFrame:
    """Build Extended Profile over exactly n_prev_sessions **before** the given session_id.

    Returns a DataFrame indexed by level_ticks with columns EP_COLS (missing if not present).
    """
    if "SessionId" not in df.columns:
        raise ValueError("df must include SessionId. Run sessions.infer_sessions first.")
    target = session_id
    prev_ids = [sid for sid in range(target - n_prev_sessions, target) if sid >= df["SessionId"].min()]
    df_slice = df[df["SessionId"].isin(prev_ids)]
    if df_slice.empty:
        return pd.DataFrame(columns=EP_COLS).astype("int64")
    ep = _count_at_levels(df_slice, tick_size)
    return ep
