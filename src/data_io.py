from __future__ import annotations
import pandas as pd
from pathlib import Path

def load_market_data(path_or_df, start_date: str, end_date: str) -> pd.DataFrame:
    """Load OHLCV-like 5m candles with optional MVC/Ask/Bid fields.

    - If `path_or_df` is a string/path, tries to read parquet or csv.

    - Time must be timezone-aware UTC or naive UTC; returns UTC tz-aware.
    """
    if isinstance(path_or_df, (str, Path)):
        p = Path(path_or_df)
        if p.suffix.lower() == ".parquet":
            df = pd.read_parquet(p)
        else:
            df = pd.read_csv(p)
    else:
        df = path_or_df.copy()

    # Ensure Time is datetime UTC tz-aware
    df["Time"] = pd.to_datetime(df["Time"], utc=True, errors="coerce")
    df = df.dropna(subset=["Time"]).sort_values("Time").reset_index(drop=True)
    # Filter by dates (inclusive)
    mask = (df["Time"] >= pd.Timestamp(start_date, tz="UTC")) & (df["Time"] <= pd.Timestamp(end_date, tz="UTC"))
    df = df.loc[mask].reset_index(drop=True)
    return df
