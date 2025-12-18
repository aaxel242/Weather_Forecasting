import numpy as np
import pandas as pd

def limpiar_datos(df):
    """
    Limpia comillas y normaliza nombres sin borrar filas vacías.
    """
    df = df.drop_duplicates()

    # Quitar comillas en todo el archivo
    df = df.applymap(lambda x: str(x).replace('"', '').strip() if pd.notnull(x) else x)

    # Normalizar nombres de columnas
    df.columns = (
        df.columns.str.strip().str.lower()
        .str.replace(" ", "_", regex=False)
        .str.replace(r"[^a-z0-9_]+", "", regex=True)
    )
    
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"]) # La fecha es lo único obligatorio
        df = df.set_index("date").sort_index()

    # Convertir columnas a numérico (float)
    for col in df.columns:
        if col != "date":
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(",", "."), errors="coerce")

    return df