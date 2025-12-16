import numpy as np
import pandas as pd
import re
import streamlit as st


def limpiar_datos(df):
    """
    Limpieza y normalización rigurosa del dataset climático,
    incluyendo la corrección de formato de comillas y espacios del CSV.
    """
    # Eliminar duplicados
    df = df.drop_duplicates()

    # Normalizar nombres de columnas (más robusto: elimina (, ), %, °, etc.)
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
        .str.replace(r"[^a-z0-9_]+", "", regex=True) # Elimina (), %, °, etc.
    )
    
    # Convertir date
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    
    # Definir columnas numéricas con los nuevos nombres limpios
    columnas_numericas = [
        "tmed", "tmin", "tmax", "prec", "dir",
        "velmedia", "racha",
        "hrmedia", "hrmax", "hrmin",
        
        # Nombres limpios después de la normalización
        "cloudcover_max", "cloudcover_min", "cloudcover_mean",
        "surface_pressure_hpa_max", "surface_pressure_hpa_min", "surface_pressure_hpa_mean",
        "dewpoint_2m_c_max", "dewpoint_2m_c_min", "dewpoint_2m_c_mean",
    ]

    # Aplicar limpieza de formato (comillas y espacios) y conversión
    for col in columnas_numericas:
        if col in df.columns:
            # Eliminar comillas dobles y espacios antes de convertir
            df[col] = (
                df[col]
                .astype(str)
                .str.replace(",", ".", regex=False)   
                .str.replace('"', '', regex=False)        
                .str.strip()                     
                .replace("Varias", np.nan)
            )
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 6. Validar rangos (este paso ahora funciona porque los datos son numéricos)
    # if "tmed" in df.columns:
    #     df = df[df["tmed"].between(-50, 60)]
    # if "hrmedia" in df.columns:
    #     df = df[df["hrmedia"].between(0, 100)]
    
    # Asegurarse de que date se use como índice si se usa para series de tiempo después
    if "date" in df.columns:
        df = df.set_index("date").sort_index()

    return df