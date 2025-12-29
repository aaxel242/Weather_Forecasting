import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error as mean_a
import joblib

# Load the processed weather dataset
dataset_weather = pd.read_csv('src/data/processed/data_weather_oficial.csv')

# convert 'date' column to datetime
dataset_weather['date'] = pd.to_datetime(dataset_weather['date'])


# Feature Engineering: Variables Temporales (Clave para patrones estacionales)
dataset_weather['mes'] = dataset_weather['date'].dt.month
dataset_weather['dia_anio'] = dataset_weather['date'].dt.dayofyear

# Select features and target variable
features = [
    # --- LOS IMPRESCINDIBLES (Térmicos) ---
    'dewpoint_2m_c_mean', 
    'dewpoint_2m_c_max', 
    'dewpoint_2m_c_min',
    
    # --- TIEMPO Y ESTACIONES (Ciclos) ---
    'dia_anio', 'mes',
    'estacion_invierno', 'estacion_verano', 
    'estacion_primavera', 'estacion_otoo',
    
    # --- ESTADO DEL CIELO ---
    'cloudcover__mean', 'cloudcover__min', 'cloudcover__max',
    'surface_pressure_hpa_max', 'surface_pressure_hpa_mean',
    
    # --- VIENTO Y OTROS ---
    'velmedia',
    'hrmax'  # La única de humedad relativa que sobrevivió al test
]

target = 'tmax'

x = dataset_weather[features]
y = dataset_weather[target]

# split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# model definition
model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs= -1)

# train the model
model.fit(X_train, y_train)

# make predictions
y_pred = model.predict(X_test)

# evaluate the model
mae = mean_a(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"MAE (Error Medio Absoluto): {mae:.4f} °C")
print(f"MSE (Error Cuadrático Medio): {mse:.4f} °C²")
print(f"RMSE (Error Cuadrático Medio): {rmse:.4f} °C")
print(f"R² (Coeficiente de Determinación): {r2:.4f}")

if r2 > 0.90:
    print("✅ RESULTADO: El modelo es EXCELENTE.")
elif r2 > 0.80:
    print("⚠️ RESULTADO: El modelo es BUENO, pero mejorable.")
else:
    print("❌ RESULTADO: Algo falla, revisa los datos.")