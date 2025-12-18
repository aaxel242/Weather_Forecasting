import pandas as pd
import numpy as np

def imputar_datos(df):
    """
    Imputa valores faltantes y redondea el resultado a 1 decimal.
    """
    df_imput = df.copy()

    # 1. Variables estacionales (Media mensual)
    cols_estacionales = [
        "tmed", "tmin", "tmax", "hrmedia", "hrmax", "hrmin",
        "dewpoint_2m_c_max", "dewpoint_2m_c_min", "dewpoint_2m_c_mean"
    ]

    for col in cols_estacionales:
        if col in df_imput.columns:
            df_imput[col] = df_imput[col].fillna(
                df_imput.groupby(df_imput.index.month)[col].transform("mean")
            )

    # 2. Precipitación (Asumir 0 si falta)
    if "prec" in df_imput.columns:
        df_imput["prec"] = df_imput["prec"].fillna(0) 

    # 3. Variables temporales (Interpolación + Respaldo media mensual)
    cols_a_interpolar = [
        "velmedia", "racha", 
        "cloudcover_max", "cloudcover_min", "cloudcover_mean",
        "surface_pressure_hpa_max", "surface_pressure_hpa_min", "surface_pressure_hpa_mean"
    ]
    
    for col in cols_a_interpolar:
        if col in df_imput.columns:
            df_imput[col] = df_imput[col].interpolate(method="linear")
            df_imput[col] = df_imput[col].fillna(
                df_imput.groupby(df_imput.index.month)[col].transform("mean")
            )

    # 4. Dirección del viento (Moda)
    if "dir" in df_imput.columns:
        moda_global = df_imput["dir"].mode()[0] if not df_imput["dir"].mode().empty else np.nan
        df_imput["dir"] = df_imput["dir"].fillna(moda_global)
      
    # 5. Relleno final y REDONDEO a 1 decimal
    df_imput = df_imput.fillna(df_imput.mean(numeric_only=True))
    
    # Aplicamos el formato de un decimal a todas las columnas numéricas
    df_imput = df_imput.round(1)

    return df_imput