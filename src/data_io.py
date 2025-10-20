from __future__ import annotations
import pandas as pd
from pathlib import Path
import datetime as dt
from typing import List, Optional
from pymongo import MongoClient
import numpy as np
from pathlib import Path
import os, yaml

def load_data_mongo(
    uri: str,
    db_name: str,
    collection_name: str,
    start_date: dt.datetime,
    end_date: dt.datetime,
    fields: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Lee velas de 5m (con Order Flow) desde Mongo, entre [start_date, end_date).

    Parámetros
    ----------
    uri, db_name, collection_name : str
        Conexión a MongoDB.
    start_date, end_date : datetime
        Rango temporal [incl, excl).
    fields : list[str] | None
        Subconjunto de campos a proyectar. Si None, usa el set por defecto.

    Returns
    -------
    DataFrame con columnas clave: 
      Time, Open, High, Low, Close, MVC, Volume, Delta, NewSession, NewWeek, NewMonth, Ask, Bid
    """
    default_fields = [
        "Time", "Open", "High", "Low", "Close",
        "MVC", "Volume", "Delta",
        "NewSession", "NewWeek", "NewMonth",
        "Ask", "Bid"
    ]
    fields = fields or default_fields
    projection = {f: 1 for f in fields}
    projection["_id"] = 0

    client = MongoClient(uri)
    col = client[db_name][collection_name]
    cursor = col.find(
        {"Time": {"$gte": start_date, "$lt": end_date}},
        projection=projection
    )
    df = pd.DataFrame(list(cursor))

    if df.empty:
        return df

    # Orden y tipos
    if "Time" in df.columns:
        df["Time"] = pd.to_datetime(df["Time"], utc=True)
        df = df.sort_values("Time").reset_index(drop=True)
    

    return df



def load_params(base_path=Path("../configs/params.yml"), local_path=Path("../configs/params.local.yml")):
    with open(base_path, "r", encoding="utf-8") as f:
        P = yaml.safe_load(f) or {}

    # override local si existe
    if local_path.exists():
        with open(local_path, "r", encoding="utf-8") as f:
            L = yaml.safe_load(f) or {}
        # merge superficial
        for k, v in L.items():
            P[k] = v if not isinstance(v, dict) else {**P.get(k, {}), **v}

    # variables de entorno (prioridad máxima)
    P.setdefault("secrets", {})
    P["secrets"]["URI"] = os.getenv("URI", P["secrets"].get("URI"))
    P["secrets"]["DB"]  = os.getenv("DB",  P["secrets"].get("DB"))
    P["secrets"]["COLLECTION"]= os.getenv("COLLECTION",P["secrets"].get("COLLECTION"))
    return P
