from __future__ import annotations
import pandas as pd

def cluster_levels(levels_df: pd.DataFrame, merge_ticks: int = 1) -> pd.DataFrame:
    """Greedy 1D clustering: merge consecutive levels within ±merge_ticks into a cluster.
    Returns a DataFrame with cluster_id, center (median), and aggregated Score.
    """
    if levels_df is None or levels_df.empty:
        return pd.DataFrame(columns=["cluster_id","center_ticks","Score_sum"])
    df = levels_df.sort_index().copy()
    clusters = []
    cur = [df.index[0]]
    for p in df.index[1:]:
        if abs(p - cur[-1]) <= merge_ticks:
            cur.append(p)
        else:
            clusters.append(cur)
            cur = [p]
    clusters.append(cur)
    rows = []
    for cid, pts in enumerate(clusters, start=1):
        sub = df.loc[pts]
        center = int(round(sub.index.median()))
        rows.append({"cluster_id": cid, "center_ticks": center, "Score_sum": float(sub["Score"].sum()), "n_levels": len(pts)})
    return pd.DataFrame(rows).set_index("cluster_id")
