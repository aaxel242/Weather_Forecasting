import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --- CONFIGURACIÓN ---
RUTA_DATOS = 'src/data/processed/data_weather_final.csv'
RUTA_MODELO = 'src/models/modelo_tmin.joblib'
RUTA_FEATURES = 'src/models/features_tmin.joblib'

def train_temp_min_model():
    print("\n--- ENTRENANDO MODELO T_MIN (Con Métricas Completas) ---")
    
    # 1. Cargar y Ordenar
    try:
        df = pd.read_csv(RUTA_DATOS)
    except FileNotFoundError:
        print(f" Error: No se encuentra {RUTA_DATOS}")
        return

    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
    
    # 2. Features y Lags
    # La mínima depende mucho de la máxima de ayer y de las nubes nocturnas
    df['tmin_yesterday'] = df['tmin'].shift(1)
    df['tmax_yesterday'] = df['tmax'].shift(1)
    
    df['mes'] = df['date'].dt.month
    df['dia_anio'] = df['date'].dt.dayofyear
    
    df = df.dropna()

    possible_features = [
        'tmin_yesterday', 'tmax_yesterday',
        'dia_anio', 'mes',
        'dewpoint_2m_c_min', 'dewpoint_2m_c_mean',
        'cloudcover__min', 'cloudcover__mean'
    ]
    features_cols = [c for c in possible_features if c in df.columns]
    
    X = df[features_cols]
    y = df['tmin']

    print(f"Features usadas: {features_cols}")

    # 3. Split (Shuffle=False)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False, random_state=42)

    # 4. Entrenamiento
    model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    # 5. Evaluación Completa
    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print("\n RESULTADOS EVALUACIÓN T_MIN:")
    print(f"   - MAE  (Error Medio Absoluto): {mae:.4f} °C")
    print(f"   - MSE  (Error Cuadrático):     {mse:.4f}")
    print(f"   - RMSE (Raíz Error Cuad.):     {rmse:.4f} °C")
    print(f"   - R²   (Precisión General):    {r2:.4f}")

    # 6. Guardado
    os.makedirs(os.path.dirname(RUTA_MODELO), exist_ok=True)
    joblib.dump(model, RUTA_MODELO)
    joblib.dump(features_cols, RUTA_FEATURES)
    print(f"\n Modelo guardado en: {RUTA_MODELO}")

if __name__ == "__main__":
    train_temp_min_model()