import numpy as np
import pandas as pd

def limpiar_datos(df):
    """
    Limpia comillas, normaliza columnas y convierte solo las columnas correctas a numéricas.
    """

    df = df.drop_duplicates()

    # Quitar comillas en todo el archivo
    df = df.map(lambda x: str(x).replace('"', '').strip() if pd.notnull(x) else x)

    # Normalizar nombres de columnas
    df.columns = (
        df.columns.str.strip().str.lower()
        .str.replace(" ", "_", regex=False)
        .str.replace(r"[^a-z0-9_]+", "", regex=True)
    )

    # Convertir fecha
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"])
        df = df.set_index("date").sort_index()

    # Columnas numéricas reales
    cols_numericas = [
        "tmed", "tmin", "tmax",
        "hrmedia", "hrmax", "hrmin",
        "dewpoint_2m_c_max", "dewpoint_2m_c_min", "dewpoint_2m_c_mean",
        "velmedia", "racha",
        "prec",
        "cloudcover_max", "cloudcover_min", "cloudcover_mean",
        "surface_pressure_hpa_max", "surface_pressure_hpa_min", "surface_pressure_hpa_mean"
    ]

    for col in cols_numericas:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(",", "."), errors="coerce")

    # Mantener dir como categórica
    if "dir" in df.columns:
        df["dir"] = df["dir"].astype(str)

    return df
