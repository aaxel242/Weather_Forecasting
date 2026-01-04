import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def train_temp_min_model(df, model_path="src/models/modelo_tmin.pkl"):
    """
    Entrena un modelo de regresión para la temperatura mínima.
    Recibe el DataFrame completo, genera lags y selecciona features óptimas.
    """
    print("\n--- ENTRENANDO MODELO T_MIN (Random Forest) ---")
    
    # 1. PREPARACIÓN DE DATOS (Feature Engineering interna)
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
    
    # Generamos Lags si no existen
    if 'tmin_yesterday' not in df.columns:
        df['tmin_yesterday'] = df['tmin'].shift(1)
        
    # Variables temporales
    df['mes'] = df['date'].dt.month
    df['dia_anio'] = df['date'].dt.dayofyear
    
    # Limpiamos nulos generados por el lag
    df = df.dropna()

    # Selección de Features Óptimas (Basado en análisis previo)
    features_list = [
        'tmin_yesterday',       # La inercia térmica (CRUCIAL)
        'dewpoint_2m_c_min',    # Relación física directa
        'dewpoint_2m_c_mean',
        'dia_anio', 'mes',      # Estacionalidad
        'estacion_invierno', 'estacion_verano',
        'cloudcover__min',      # Nubes nocturnas (evitan heladas)
        'cloudcover__mean'
    ]
    
    # Filtramos para asegurar que solo usamos columnas que existen
    features_cols = [c for c in features_list if c in df.columns]
    
    X = df[features_cols]
    y = df['tmin']

    # 2. SPLIT (Respetando orden temporal)
    split = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    print(f"Features usadas: {features_cols}")
    print(f"Datos de entrenamiento: {len(X_train)} | Test: {len(X_test)}")

    # 3. PIPELINE Y ENTRENAMIENTO
    model = Pipeline([
        ('scaler', StandardScaler()), # Normalizamos (buena práctica aunque sea RF)
        ('regressor', RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1))
    ])

    # Entrenamos
    model_pipeline.fit(X_train, y_train)
    
    # 4. EVALUACIÓN
    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print(f"📊 RESULTADOS T_MIN:")
    print(f"   - MAE (Error Medio): {mae:.4f} °C")
    print(f"   - RMSE: {rmse:.4f} °C")
    print(f"   - R² Score: {r2:.4f}")

    # 5. GUARDADO
    # Usamos ruta absoluta para evitar errores
    base_dir = os.getcwd()
    abs_model_path = os.path.join(base_dir, model_path)
    
    os.makedirs(os.path.dirname(abs_model_path), exist_ok=True)
    joblib.dump(model, abs_model_path)
    print(f"✅ Modelo guardado en: {abs_model_path}")
    
    predictions = pd.Series(y_pred, index=y_test.index)
    return model, predictions

# Bloque para ejecutar este script directamente si se desea
if __name__ == "__main__":
    # Carga de prueba
    try:
        df_load = pd.read_csv('src/data/processed/data_weather_oficial.csv')
        train_temp_min_model(df_load)
    except FileNotFoundError:
        print("⚠️ No se encontró el dataset para prueba rápida.")
