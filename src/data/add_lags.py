import pandas as pd
import numpy as np

def add_lag_features(df):
    """
    Añade variables de retraso (lags) y targets al DataFrame.
    """
    # Trabajamos sobre una copia para no alterar el original
    df = df.copy()

    # Si el índice es la fecha  la reseteamos para operar
    if df.index.name == 'date':
        df = df.reset_index()

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # 0. Crear columna binaria si no existe
    if 'bin_prep' not in df.columns and 'prec' in df.columns:
        df['bin_prep'] = (df['prec'] > 0).astype(int)

    # 1. ESTACIONALIDAD
    df["mes"] = df["date"].dt.month
    df["dia_del_anio"] = df["date"].dt.dayofyear
    df["semana"] = df["date"].dt.isocalendar().week.astype(int)
    df["es_fin_de_semana"] = (df["date"].dt.weekday >= 5).astype(int)

    # 2. LAGS (Pasado)
    lags = [1, 2, 3, 7]
    for lag in lags:
        if "tmax" in df.columns: df[f"temp_max_lag{lag}"] = df["tmax"].shift(lag)
        if "tmin" in df.columns: df[f"temp_min_lag{lag}"] = df["tmin"].shift(lag)
        if "prec" in df.columns: df[f"precipitacion_lag{lag}"] = df["prec"].shift(lag)

    # 3. ROLLING WINDOWS
    if "tmax" in df.columns: df["temp_max_rolling_7"] = df["tmax"].rolling(7).mean()
    if "tmin" in df.columns: df["temp_min_rolling_7"] = df["tmin"].rolling(7).mean()
    if "prec" in df.columns: df["precip_rolling_7"] = df["prec"].rolling(7).sum()

    # 4. TARGETS A 7 DÍAS (Lo que queremos predecir)
    df["temp_max_target"] = df["tmax"].shift(-7)
    df["temp_min_target"] = df["tmin"].shift(-7)
    df["precipitacion_target"] = df["prec"].shift(-7)
    df["lluvia_binaria_target"] = df["bin_prep"].shift(-7)

    # 5. IMPUTACIÓN DE HUECOS (Creados por los lags)
    df = df.infer_objects(copy=False)
    df = df.interpolate(method="linear")
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].mean())

    # 6. ELIMINAR FILAS SIN TARGET
    # (Las últimas 7 filas no tienen 'mañana', por eso son nulas y las quitamos)
    df = df.dropna(subset=["temp_min_target", "lluvia_binaria_target"]).reset_index(drop=True)

    return df # <--- CAMBIO: Devolvemos el DataFrame procesado

# IMPORTANTE: No pongas código suelto aquí abajo. 
# Si quieres hacer pruebas, usa este bloque:
if __name__ == "__main__":
    print("Este código solo se ejecuta si lanzas add_lags.py directamente.")
