import joblib
import pandas as pd
import os

from src.utils.limpieza import limpiar_datos
from src.utils.imputar_datos import imputar_datos

from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# --- CONFIGURACIÓN ---
RUTA_DATOS = 'src/data/processed/data_weather_final.csv'
RUTA_MODELO = 'src/models/modelo_lluvia.joblib'
RUTA_FEATURES = 'src/models/features_lluvia.joblib'

def imprimir_metricas(y_test, y_pred, nombre_modelo):
    # Calcula e imprime métricas de clasificación: Recall, Precisión, Accuracy, F1 y Matriz de Confusión.
    # Parámetros: y_test (valores reales), y_pred (predicciones), nombre_modelo (etiqueta para mostrar).
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    print(f"\n📊 RESULTADOS {nombre_modelo}:")
    print(f"   - Recall:    {rec:.4f}")
    print(f"   - Precision: {prec:.4f}")
    print(f"   - Accuracy:  {acc:.4f}")
    print(f"   - F1-Score:  {f1:.4f}")
    print(f"\n   Matriz de Confusión:")
    print(f"   TN: {cm[0][0]} | FP: {cm[0][1]}")
    print(f"   FN: {cm[1][0]} | TP: {cm[1][1]}")

def train_rain_model_optimized():
    # Entrena un modelo RandomForest con SMOTE para clasificar días lluviosos vs secos.
    # Utiliza características como delta de presión (indicador de tormentas) y humedad.
    # Retorna modelo entrenado y lo guarda en disco.
    print("\n--- ENTRENANDO MODELO LLUVIA (OPTIMIZADO: Delta Presión + Umbral) ---")
    
    try:
        df = pd.read_csv(RUTA_DATOS)
    except FileNotFoundError:
        print(f" Error: No se encuentra {RUTA_DATOS}")
        return
    
    df = limpiar_datos(df)
    df = imputar_datos(df)

    if 'date' in df.columns:
        # convierte la columna date en datetime y ordena cornologicamente
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')

    # A. Lags Clásicos
    df['rain_yesterday_bin'] = (df['precipitacion_lag1'] > 0.1).astype(int)
    
    # B. NUEVO: Tendencia de Presión (Pressure Delta)
    # La caída de presión es el mejor predictor físico de tormentas
    df['pressure_yesterday'] = df['surface_pressure_hpa_mean'].shift(1)
    df['pressure_delta'] = df['surface_pressure_hpa_mean'] - df['pressure_yesterday'] 
    # (Si es negativo significa que la presión está cayendo)
    
    df = df.dropna(subset=['pressure_delta'])

    features_cols = [
        'precipitacion_lag1', 'rain_yesterday_bin', 'pressure_delta', 'surface_pressure_hpa_mean', 
        'cloudcover__mean', 'cloudcover__max','hrmedia', 'hrmax','dewpoint_2m_c_mean','mes', 'dia_del_anio',
        'estacion_invierno', 'estacion_verano', 'estacion_otoo','estacion_primavera'
    ]

    # Usamos la técnica de seguridad (Copy + Check NaNs)
    df_model = df[features_cols + ['bin_prep']].copy()
    
    if df_model.isna().any().any():
        print("⚠️ Advertencia: Aún hay NaNs. Aplicando limpieza de emergencia.")
        df_model = df_model.dropna()

    x = df_model[features_cols]
    y = df_model['bin_prep']

    print(f"Features usadas: {features_cols}")

    # 3. SPLIT
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=False, random_state=42)

    # 4. ENTRENAMIENTO
    # Aumentamos n_estimators y limitamos profundidad para evitar memorización
    pipeline_rf = Pipeline([
        ('smote', SMOTE(random_state=42)),
        ('rf', RandomForestClassifier(
            random_state=42,
        ))
    ])

    param_grid = {
        'rf__n_estimators': [500],
        'rf__max_depth': [8],
        'rf__min_samples_leaf': [10]
    }

    grid_rf = GridSearchCV(pipeline_rf, param_grid, cv=5, scoring='f1', n_jobs=-1)
    grid_rf.fit(X_train, y_train)

    y_pred = grid_rf.best_estimator_.predict(X_test)

    imprimir_metricas(y_test, y_pred, "RANDOM FOREST")
    # NOTA IMPORTANTE:
    # Random Forest standard de sklearn no permite guardar el umbral dentro del objeto .pkl.
    # El umbral se aplica en la lógica de predicción.
    # Aquí guardamos el modelo tal cual, pero en 'prediction_engine.py' aplicaremos el truco.

    # 6. Guardado
    os.makedirs(os.path.dirname(RUTA_MODELO), exist_ok=True)
    joblib.dump(grid_rf.best_estimator_, RUTA_MODELO)
    joblib.dump(features_cols, RUTA_FEATURES)
    print(f"\n Modelo guardado en: {RUTA_MODELO}")

if __name__ == "__main__":
    train_rain_model_optimized()