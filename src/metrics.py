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


def hits_exposure_for_levels(df_session, levels_ticks, tick_size, X_ticks=1,
                             refractory_min=10, decay=0.6):
    """
    - Cuenta hits con ventana refractaria (minutos) para no sobrecontar.
    - Aplica decaimiento a partir del 2º toque en la misma sesión.
    Devuelve:
      hits_eff: Serie (ponderada por decay)
      exposure: Serie (minutos/velas 'cerca')
      hits_raw: Serie (conteo simple de toques)
    """

    if len(df_session) == 0 or len(levels_ticks) == 0:
        z = pd.Series(dtype="float64")
        return z, z, z

    ts = pd.to_datetime(df_session["Time"].values)
    hi = np.round(df_session["High"].values / tick_size).astype("int64")
    lo = np.round(df_session["Low"].values  / tick_size).astype("int64")
    center = np.round(((df_session["Open"].values + df_session["Close"].values) / 2.0) / tick_size).astype("int64")

    levels_ticks = pd.Index(levels_ticks.astype("int64"), name="level_ticks")
    hits_eff = pd.Series(0.0, index=levels_ticks)
    hits_raw = pd.Series(0,   index=levels_ticks, dtype="int64")
    expo     = pd.Series(0,   index=levels_ticks, dtype="int64")

    # precomputo exposición por vela a cada banda
    for p in levels_ticks:
        band_lo, band_hi = p - X_ticks, p + X_ticks
        near = (center >= band_lo) & (center <= band_hi)
        expo.loc[p] = int(near.sum())

        # toques crudos por vela
        touched = (lo <= band_hi) & (hi >= band_lo)
        if not touched.any():
            continue

        # aplicar refractario y decay
        last_hit_time = None
        k = 0
        for i, is_hit in enumerate(touched):
            if not is_hit:
                continue
            t_i = ts[i]
            if last_hit_time is None or (t_i - last_hit_time).total_seconds() >= refractory_min*60:
                k += 1
                hits_eff.loc[p] += decay**(k-1)
                hits_raw.loc[p] += 1
                last_hit_time = t_i

    return hits_eff, expo, hits_raw



def cp_from_ep_row(ep_row):
    """
    CP (pureza de confluencia): nº de familias presentes / total posibles.
    Familias consideradas: High, Low, Close, MVC, UpperWick, LowerWick.
    """
    fams = ["High","Low","Close","MVC","UpperWick","LowerWick"]
    present = sum(1 for f in fams if f in ep_row.index and ep_row.get(f, 0) > 0)
    return present / float(len(fams))

def dense_hue_map(df_session, tick_size, tick_min, tick_max, X_ticks=1,
                  refractory_min=10, decay=0.6):
    """
    Calcula HUE en TODOS los ticks del rango [tick_min, tick_max].
    Útil para Contrast Lift: comparamos cada baliza con su vecindad densa.
    """

    ticks_grid = pd.Index(range(int(tick_min), int(tick_max)+1), name="level_ticks")
    hits_eff, expo, _ = hits_exposure_for_levels(
        df_session, ticks_grid, tick_size, X_ticks=X_ticks,
        refractory_min=refractory_min, decay=decay
    )
    HUE_dense = (hits_eff / expo.replace(0,1)).rename("HUE_dense")
    return HUE_dense

