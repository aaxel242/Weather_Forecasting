import pandas as pd

def add_binari_precipitation():
    """
    Agrega una columna binaria 'bin_prep' al DataFrame indicando si hubo precipitación.

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame que contiene la columna 'prec'.

    Retorna
    -------
    pd.DataFrame
        DataFrame con la nueva columna 'bin_prep'.
    """
    # Cargar el dataset
    df = pd.read_csv('src/data/processed/data_weather_oficial.csv')

    # Crear la columna binaria: 1 si prec > 0, de lo contrario 0
    df['bin_prep'] = (df['prec'] > 0).astype(int)

    # Guardar o ver resultado
    print(df[['date', 'prec', 'bin_prep']].head())
    df.to_csv('src/data/raw/data_binario.csv', index=False)

def add_lag_features():

    df = pd.read_csv("src/data/raw/data_binario.csv")

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # ============================
    # 1. ESTACIONALIDAD
    # ============================
    df["mes"] = df["date"].dt.month
    df["dia_del_anio"] = df["date"].dt.dayofyear
    df["semana"] = df["date"].dt.isocalendar().week.astype(int)
    df["es_fin_de_semana"] = (df["date"].dt.weekday >= 5).astype(int)

    # ============================
    # 2. LAGS
    # ============================
    lags = [1, 2, 3, 7]
    for lag in lags:
        df[f"temp_max_lag{lag}"] = df["tmax"].shift(lag)
        df[f"temp_min_lag{lag}"] = df["tmin"].shift(lag)
        df[f"precipitacion_lag{lag}"] = df["prec"].shift(lag)

    # ============================
    # 3. ROLLING WINDOWS
    # ============================
    df["temp_max_rolling_7"] = df["tmax"].rolling(7).mean()
    df["temp_min_rolling_7"] = df["tmin"].rolling(7).mean()
    df["precip_rolling_7"] = df["prec"].rolling(7).sum()

    # ============================
    # 4. TARGETS A 7 DÍAS
    # ============================
    df["temp_max_target"] = df["tmax"].shift(-7)
    df["temp_min_target"] = df["tmin"].shift(-7)
    df["precipitacion_target"] = df["prec"].shift(-7)
    df["lluvia_binaria_target"] = df["bin_prep"].shift(-7)

    # ============================
    # 5. IMPUTACIÓN DE FEATURES
    # ============================
    df = df.interpolate(method="linear")
    df = df.fillna(df.mean(numeric_only=True))

    # Booleanos
    df["es_fin_de_semana"] = df["es_fin_de_semana"].fillna(0).astype(int)

    # ============================
    # 6. ELIMINAR SOLO NULOS DE TARGETS
    # ============================
    df = df.dropna(subset=[
        "temp_max_target",
        "temp_min_target",
        "precipitacion_target",
        "lluvia_binaria_target"
    ]).reset_index(drop=True)

    df.to_csv("src/data/processed/data_weather_final.csv", index=False)

df = pd.read_csv("src/data/processed/data_weather_final.csv")
df = df.drop(columns=["precipitacion_target"])
df.to_csv("src/data/processed/data_weather_final.csv", index=False)