import numpy as np
import pandas as pd

def limpiar_datos(df):
    """
    Limpia comillas, normaliza columnas y asegura tipos numéricos compatibles.
    """
    # 1. Eliminar duplicados
    df = df.drop_duplicates()

    # 2. Limpieza de strings y comillas
    df = df.map(lambda x: str(x).replace('"', '').strip() if pd.notnull(x) else x)

    # 3. Normalizar nombres de columnas
    df.columns = (
        df.columns.str.strip().str.lower()
        .str.replace(" ", "_", regex=False)
        .str.replace(r"[^a-z0-9_]+", "", regex=True)
    )

    # 4. Convertir fecha y establecer índice
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"])
        df = df.set_index("date").sort_index()

        # Asegúrate de que esto esté así en tu archivo de limpieza:
        # df["date"] = pd.to_datetime(df["date"], errors="coerce")
        # df = df.dropna(subset=["date"])
        # df = df.set_index("date").sort_index() # El sort_index es clave para la línea temporal

    # 5. Lista de columnas que DEBEN ser numéricas (incluimos 'dir')
    cols_numericas = [
        "tmed", "prec", "tmin", "tmax", "dir",  # 'dir' ahora es numérica
        "velmedia", "racha", "hrmedia", "hrmax", "hrmin",
        "cloudcover__max", "cloudcover__min", "cloudcover__mean",
        "surface_pressure_hpa_max", "surface_pressure_hpa_min", "surface_pressure_hpa_mean",
        "dewpoint_2m_c_max", "dewpoint_2m_c_min", "dewpoint_2m_c_mean",
        "estacion_invierno", "estacion_otoo", "estacion_primavera", "estacion_verano",
        "bin_prep", "mes", "dia_del_anio", "semana", "es_fin_de_semana",
        "temp_max_lag1", "temp_min_lag1", "precipitacion_lag1",
        "temp_max_lag2", "temp_min_lag2", "precipitacion_lag2",
        "temp_max_lag3", "temp_min_lag3", "precipitacion_lag3",
        "temp_max_lag7", "temp_min_lag7", "precipitacion_lag7",
        "temp_max_target", "temp_min_target", "precipitacion_target"
    ]

    for col in cols_numericas:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(",", "."), errors="coerce")

    return df
