import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os

# --- CONFIGURACIÓN ---
RUTA_DATOS = 'src/data/processed/data_weather_final.csv'
RUTA_MODELO = 'src/models/modelo_tmax.pkl'
RUTA_FEATURES = 'src/models/features_tmax.pkl'

def train_tmax():
    print("--- ENTRENANDO MODELO T_MAX (Con Lags y Métricas Completas) ---")
    
    # 1. Cargar Datos
    try:
        df = pd.read_csv(RUTA_DATOS)
    except FileNotFoundError:
        print(f"❌ Error: No se encuentra {RUTA_DATOS}")
        return

    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date') # Vital para Time Series

    # 2. Ingeniería de Características (Lags)
    # Creamos "Ayer" para predecir "Hoy"
    df['tmax_yesterday'] = df['tmax'].shift(1)
    df['tmin_yesterday'] = df['tmin'].shift(1)
    
    # Datos temporales
    df['mes'] = df['date'].dt.month
    df['dia_anio'] = df['date'].dt.dayofyear
    
    # Eliminamos la primera fila (que tiene NaN por el shift)
    df = df.dropna()

    # Selección de Features (Solo las que existan en el CSV)
    possible_features = [
        'tmax_yesterday', 'tmin_yesterday',
        'dia_anio', 'mes',
        'dewpoint_2m_c_mean', 'dewpoint_2m_c_max',
        'cloudcover__mean', 'velmedia', 'hrmax'
    ]
    features = [c for c in possible_features if c in df.columns]
    
    X = df[features]
    y = df['tmax']

    print(f"Features usadas: {features}")

    # 3. Split (Sin barajar - Time Series Split)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False, random_state=42)

    # 4. Entrenamiento
    model = RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=42)
    model.fit(X_train, y_train)

    # 5. Evaluación Completa
    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print("\n📊 RESULTADOS EVALUACIÓN T_MAX:")
    print(f"   - MAE  (Error Medio Absoluto): {mae:.4f} °C")
    print(f"   - MSE  (Error Cuadrático):     {mse:.4f}")
    print(f"   - RMSE (Raíz Error Cuad.):     {rmse:.4f} °C")
    print(f"   - R²   (Precisión General):    {r2:.4f} (1.0 es perfecto)")

    # 6. Guardado
    os.makedirs(os.path.dirname(RUTA_MODELO), exist_ok=True)
    joblib.dump(model, RUTA_MODELO)
    joblib.dump(features, RUTA_FEATURES) # Guardamos la lista de columnas
    print(f"\n✅ Modelo guardado en: {RUTA_MODELO}")

if __name__ == "__main__":
    train_tmax()