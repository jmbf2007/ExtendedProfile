from __future__ import annotations
import numpy as np
import pandas as pd

def cluster_levels(levels_df: pd.DataFrame, merge_ticks: int = 1, score_col: str = "Score",
                   center_method: str = "median") -> pd.DataFrame:
    """
    Greedy 1D clustering: agrupa niveles consecutivos dentro de ±merge_ticks.
    Devuelve un DataFrame con:
      - cluster_id
      - center_ticks (mediana o media ponderada por score del cluster)
      - Score_sum (suma de la columna score_col en el cluster)
      - n_levels (nº de niveles en el cluster)

    Parámetros:
      - score_col: "Score" (por defecto) o "Score_star" si trabajas crowd-aware.
      - center_method: "median" | "wavg" (media ponderada por score_col).
    """
    if levels_df is None or levels_df.empty:
        return pd.DataFrame(columns=["cluster_id","center_ticks","Score_sum","n_levels"]).set_index("cluster_id")

    # Asegura índice entero (ticks)
    df = levels_df.copy()
    try:
        df.index = df.index.astype("int64")
    except Exception:
        df.index = pd.Index(df.index.values, dtype="int64")

    df = df.sort_index()

    # Construye grupos adyacentes
    clusters = []
    cur = [df.index[0]]
    for p in df.index[1:]:
        if abs(int(p) - int(cur[-1])) <= merge_ticks:
            cur.append(int(p))
        else:
            clusters.append(cur)
            cur = [int(p)]
    clusters.append(cur)

    rows = []
    for cid, pts in enumerate(clusters, start=1):
        sub = df.loc[pts]
        # Centro del cluster
        if center_method == "wavg" and score_col in sub.columns and sub[score_col].sum() > 0:
            center_val = np.average(sub.index.values, weights=sub[score_col].values)
        else:
            center_val = np.median(sub.index.values)
        center = int(round(center_val))

        # Suma de scores
        score_sum = float(sub[score_col].sum()) if score_col in sub.columns else float(sub["Score"].sum())

        rows.append({
            "cluster_id": cid,
            "center_ticks": center,
            "Score_sum": score_sum,
            "n_levels": len(pts),
        })

    return pd.DataFrame(rows).set_index("cluster_id")
