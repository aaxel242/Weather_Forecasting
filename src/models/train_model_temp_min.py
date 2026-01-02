import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

def train_temp_min_model(data_path='src/data/processed/data_weather_final.csv', model_path="src/models/model_temp_min_rf.pkl"):
    # 1. Cargamos datos
    df = pd.read_csv(data_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date') # Crucial para series temporales

    # 2. Ingeniería de Características
    df['mes'] = df['date'].dt.month
    df['dia_anio'] = df['date'].dt.dayofyear
    
    # 3. Selección de Features
    features_list = [
        'dewpoint_2m_c_mean', 'dewpoint_2m_c_max', 'dewpoint_2m_c_min',
        'dia_anio', 'mes',
        'estacion_invierno', 'estacion_verano', 'estacion_primavera', 'estacion_otoo',
        'cloudcover__mean', 'cloudcover__min', 'cloudcover__max',
        'surface_pressure_hpa_max', 'surface_pressure_hpa_mean',
        'velmedia', 'hrmax'
    ]
    
    target = 'tmin' 
    
    X = df[features_list]
    y = df[target]

    # 4. División Cronológica 
    split = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    # 5. Pipeline con Escalado 
    model_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1))
    ])

    # Entrenamos
    model_pipeline.fit(X_train, y_train)
    
    # 6. Guardamos el modelo
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model_pipeline, model_path)
    
    # 7. Evaluación Completa (Basado en el modelo de Máxima)
    predictions = model_pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    print(f"--- Evaluacion Modelo Temp Minima ---")
    print(f"MAE (Error Medio Absoluto): {mae:.4f} °C")
    print(f"RMSE (Error Cuadrático Medio): {rmse:.4f} °C")
    print(f"R² (Coeficiente de Determinación): {r2:.4f}")

    print(f"MSE (Error Cuadrático Medio): {mse:.4f} °C²") # para que la computadora aprenda ( castiga los fallos graves)
    
    return model_pipeline, predictions

if __name__ == "__main__":
    train_temp_min_model()
