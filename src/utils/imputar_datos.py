def imputar_datos(df):
    """
    Imputa valores faltantes sin romper tipos de datos.
    """
    df_imput = df.copy()

    # 1. Variables estacionales (media mensual)
    cols_estacionales = [
        "tmed", "tmin", "tmax",
        "hrmedia", "hrmax", "hrmin",
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

    # 4. Dirección del viento (moda)
    if "dir" in df_imput.columns:
        moda_global = df_imput["dir"].mode()[0] if not df_imput["dir"].mode().empty else "N"
        df_imput["dir"] = df_imput["dir"].fillna(moda_global)

    # 5. Relleno final numérico
    df_imput = df_imput.fillna(df_imput.mean(numeric_only=True))

    # 6. Redondeo
    df_imput = df_imput.round(1)

    return df_imput

# import pandas as pd
# import numpy as np


# def imputar_datos(df):
#     df_imput = df.copy()
    
#     # Asegurarnos que la columna de fecha existe para agrupar
#     if 'date' in df_imput.columns:
#         df_imput['date'] = pd.to_datetime(df_imput['date'])
#         meses = df_imput['date'].dt.month
#     else:
#         # Si la fecha es el índice
#         meses = df_imput.index.month

#     # Columnas numéricas (todas las que mencionaste)
#     cols_num = df_imput.select_dtypes(include=[np.number]).columns

#     for col in cols_num:
#         # Imputar por media del mes (estacionalidad)
#         df_imput[col] = df_imput[col].fillna(
#             df_imput.groupby(meses)[col].transform("mean")
#         )
    
#     # Relleno final para casos extremos y redondeo
#     df_imput = df_imput.fillna(df_imput.mean(numeric_only=True)).round(2)
#     return df_imput