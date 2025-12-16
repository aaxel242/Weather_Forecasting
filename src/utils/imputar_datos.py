import pandas as pd
import numpy as np

def imputar_datos(df):
    """
    Imputa valores faltantes en el dataset climático utilizando:
    1. Media condicional (por mes) para variables estacionales (temperaturas, humedad).
    2. Interpolación lineal para la mayoría de series temporales.
    3. Cero para precipitación y moda para dirección de viento.
    """
    # Se espera que 'df' ya esté limpio y con 'date' como índice (como en tu proceso anterior)
    df_imput = df.copy()

    # Estrategia para Variables Estacionales (Temperatura y Humedad) ---
    # La media de todo el dataset es engañosa. Usamos la MEDIA MENSUAL.
    # Calcular la media de cada columna agrupada por mes
    media_mensual = df_imput.groupby(df_imput.index.month).mean()

    # Columnas que se benefician de la media estacional
    cols_estacionales = ["tmed", "tmin", "tmax", "hrmedia", "hrmax", "hrmin",
                         "dewpoint_2m_c_max", "dewpoint_2m_c_min", "dewpoint_2m_c_mean"]

    print("Imputando variables estacionales (Media Condicional por Mes)...")
    for col in cols_estacionales:
        if col in df_imput.columns:
            # Rellenar los NaNs en cada mes con la media de ese mes
            df_imput[col] = df_imput[col].fillna(
                df_imput.groupby(df_imput.index.month)[col].transform("mean")
            )

    # Si aún quedan NaNs (e.g., si todo un mes está vacío), usamos la interpolación (ver paso 3)

    # Usamos 0, ya que es un valor binario/discreto (lluvia o no)
    if "prec" in df_imput.columns:
        df_imput["prec"] = df_imput["prec"].fillna(0) 

    # Para la mayoría de las variables continuas y el viento, la interpolación lineal es excelente.
    # Rellena el NaN usando los valores de los días inmediatamente anteriores y posteriores. 
    cols_a_interpolar = [
        "velmedia", "racha", 
        "cloudcover_max", "cloudcover_min", "cloudcover_mean",
        "surface_pressure_hpa_max", "surface_pressure_hpa_min", "surface_pressure_hpa_mean"
    ]
    
    print("Imputando variables temporales (Interpolación Lineal)...")
    for col in cols_a_interpolar:
        if col in df_imput.columns:
            # Primero intenta la interpolación lineal (útil si faltan 1 o 2 días)
            df_imput[col] = df_imput[col].interpolate(method="linear")
            
            # Luego, si aún quedan NaNs (si faltan grandes bloques), usa la media mensual como respaldo
            df_imput[col] = df_imput[col].fillna(
                df_imput.groupby(df_imput.index.month)[col].transform("mean")
            )
            # Finalmente, si aún hay NaNs, usa la media global (como último recurso)
            df_imput[col] = df_imput[col].fillna(df_imput[col].mean())


    # Variable circular/categórica: La moda es la opción más segura.
    if "dir" in df_imput.columns:
        moda_global = df_imput["dir"].mode()[0] if not df_imput["dir"].mode().empty else np.nan
        df_imput["dir"] = df_imput["dir"].fillna(moda_global)
      
    # Por si se ha colado algún NaN en otras columnas, se rellena con la media global.
    df_imput = df_imput.fillna(df_imput.mean(numeric_only=True))

    return df_imput