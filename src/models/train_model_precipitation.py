import joblib
import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# --- CONFIGURACIÓN ---
RUTA_DATOS = 'src/data/processed/data_weather_final.csv'
RUTA_MODELO = 'src/models/modelo_lluvia.joblib'
RUTA_FEATURES = 'src/models/features_lluvia.joblib'

def train_rain_model_optimized():
    print("\n--- ENTRENANDO MODELO LLUVIA (OPTIMIZADO: Delta Presión + Umbral) ---")
    
    try:
        df = pd.read_csv(RUTA_DATOS)
    except FileNotFoundError:
        print(f"❌ Error: No se encuentra {RUTA_DATOS}")
        return

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
    
    # df = df.dropna()

    features_cols = [
        'precipitacion_lag1', 'rain_yesterday_bin',
        'pressure_delta', 
        'surface_pressure_hpa_mean', 
        'cloudcover__mean', 'cloudcover__max',
        'hrmedia', 'hrmax',
        'dewpoint_2m_c_mean',
        'mes', 'dia_del_anio'
    ]

    x = df[features_cols]
    y = df['bin_prep']

    print(f"Features usadas: {features_cols}")

    # 3. SPLIT
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=False, random_state=42)

    # 4. ENTRENAMIENTO
    # Aumentamos n_estimators y limitamos profundidad para evitar memorización
    model = RandomForestClassifier(
        n_estimators=500, 
        max_depth=20,      
        class_weight='balanced',
        random_state=42, 
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    # 5. EVALUACIÓN CON UMBRAL AJUSTADO (Threshold Tuning)
    # En lugar de usar predict() directo (que corta en 0.5), usamos predict_proba()
    # Si la probabilidad de lluvia es > 0.26 (26%), predecimos Lluvia.
    
    y_probs = model.predict_proba(X_test)[:, 1] # Probabilidad de clase 1 (Lluvia)
    UMBRAL_OPTIMO = 0.26  # <--- Hacemos al modelo más sensible
    
    y_pred_ajustado = (y_probs >= UMBRAL_OPTIMO).astype(int)
    
    # Métricas
    acc = accuracy_score(y_test, y_pred_ajustado)
    prec = precision_score(y_test, y_pred_ajustado, zero_division=0)
    rec = recall_score(y_test, y_pred_ajustado, zero_division=0)
    f1 = f1_score(y_test, y_pred_ajustado, zero_division=0)
    
    print(f"\n📊 RESULTADOS CON UMBRAL {UMBRAL_OPTIMO} (Más sensible):")
    print(f"   - Recall (Sensibilidad): {rec:.4f} (¡Buscamos que esto suba!)")
    print(f"   - Precision:             {prec:.4f}")
    print(f"   - Accuracy:              {acc:.4f}")
    print(f"   - F1-Score:              {f1:.4f}")

    # Matriz de Confusión
    cm = confusion_matrix(y_test, y_pred_ajustado)
    print(f"\n   Matriz de Confusión Ajustada:")
    print(f"   TN (Sol ok): {cm[0][0]} | FP (Falsa alarma): {cm[0][1]}")
    print(f"   FN (Lluvia perdida): {cm[1][0]} | TP (Lluvia detectada): {cm[1][1]}")

    # NOTA IMPORTANTE:
    # Random Forest standard de sklearn no permite guardar el umbral dentro del objeto .pkl.
    # El umbral se aplica en la lógica de predicción.
    # Aquí guardamos el modelo tal cual, pero en 'prediction_engine.py' aplicaremos el truco.

    # 6. Guardado
    os.makedirs(os.path.dirname(RUTA_MODELO), exist_ok=True)
    joblib.dump(model, RUTA_MODELO)
    joblib.dump(features_cols, RUTA_FEATURES)
    print(f"\n✅ Modelo guardado en: {RUTA_MODELO}")

if __name__ == "__main__":
    train_rain_model_optimized()