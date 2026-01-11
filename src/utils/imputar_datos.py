import pandas as pd

def imputar_datos(df):
    """
    Imputa valores faltantes con lógica de arrastre para 'dir' 
    y estacionalidad para el resto.
    """
    df_imput = df.copy()

    # Asegurar que el índice sea datetime para que .index.month funcione
    if not isinstance(df_imput.index, pd.DatetimeIndex):
        if 'date' in df_imput.columns:
            df_imput['date'] = pd.to_datetime(df_imput['date'])
            df_imput.set_index('date', inplace=True)

    # 1. Variables estacionales (media mensual)
    cols_estacionales = [
        "tmed", "tmin", "tmax", "hrmedia", "hrmax", "hrmin",
        "dewpoint_2m_c_max", "dewpoint_2m_c_min", "dewpoint_2m_c_mean"
    ]
    for col in cols_estacionales:
        if col in df_imput.columns:
            df_imput[col] = df_imput[col].fillna(
                df_imput.groupby(df_imput.index.month)[col].transform("mean")
            )

    # 2. Precipitación (asumir 0 si falta)
    if "prec" in df_imput.columns:
        df_imput["prec"] = df_imput["prec"].fillna(0)

    # 3. Variables temporales (interpolación + media mensual)
    cols_a_interpolar = [
        "velmedia", "racha", "cloudcover_max", "cloudcover_min", "cloudcover_mean",
        "surface_pressure_hpa_max", "surface_pressure_hpa_min", "surface_pressure_hpa_mean"
    ]
    for col in cols_a_interpolar:
        if col in df_imput.columns:
            df_imput[col] = df_imput[col].interpolate(method="linear")
            df_imput[col] = df_imput[col].fillna(
                df_imput.groupby(df_imput.index.month)[col].transform("mean")
            )

    # 4. Dirección del viento (Lógica solicitada: Dato anterior -> Media)
    # En imputar_datos:
    if "dir" in df_imput.columns:
        # Arrastra el valor anterior (Ej: 28.0 -> 28.0)
        df_imput["dir"] = df_imput["dir"].ffill()
        # Si sigue habiendo nulos al principio, usa la media
        df_imput["dir"] = df_imput["dir"].fillna(df_imput["dir"].mean())

    # 5. Relleno final numérico y redondeo
    df_imput = df_imput.fillna(df_imput.mean(numeric_only=True))

    return df_imput
