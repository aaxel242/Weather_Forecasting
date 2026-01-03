import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os

# CONFIGURACIÓN
RUTA_DATOS = 'src/data/processed/data_weather_oficial.csv'
RUTA_MODELO = 'src/models/modelo_tmax.pkl'

print("--- ENTRENANDO MODELO T_MAX ---")
df = pd.read_csv(RUTA_DATOS)
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date')

# Feature Engineering (Lags + Fechas)
df['tmax_yesterday'] = df['tmax'].shift(1)
df['mes'] = df['date'].dt.month
df['dia_anio'] = df['date'].dt.dayofyear
df = df.dropna()

features = [
    'tmax_yesterday', 'dewpoint_2m_c_mean', 'dewpoint_2m_c_max',
    'dia_anio', 'mes', 'estacion_invierno', 'estacion_verano',
    'cloudcover__mean', 'velmedia', 'hrmax'
]
target = 'tmax'

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=42)
model.fit(X_train, y_train)

# Evaluación
y_pred = model.predict(X_test)
print(f"MAE: {mean_absolute_error(y_test, y_pred):.4f} °C")
print(f"MSE: {mean_squared_error(y_test, y_pred):.4f} °C²")
print(f"R2:  {r2_score(y_test, y_pred):.4f}")

os.makedirs(os.path.dirname(RUTA_MODELO), exist_ok=True)
joblib.dump(model, RUTA_MODELO)
print(f"Modelo guardado en {RUTA_MODELO}")