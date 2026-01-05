import joblib
import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def preparar_datos_lluvia(df):
    """Auxiliar para preparar features de lluvia"""

    df = df.copy()

    # --- GENERACIÓN DE VARIABLES (CORRECCIÓN IMPORTANTE) ---
    # Si no creamos esto, el modelo no encuentra 'mes' ni 'dia_anio' y falla luego
    # -------------------------------------------------------
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        df['mes'] = df['date'].dt.month
        df['dia_anio'] = df['date'].dt.dayofyear

    # Crear target binario si no existe
    if 'bin_prep' not in df.columns and 'prec' in df.columns:
        df['bin_prep'] = (df['prec'] > 0).astype(int)
    
    # --- FEATURES RECOMENDADAS ---
    # He añadido dewpoint y lags que son vitales para la lluvia
    features_list = [
        'cloudcover__mean', 'cloudcover__max', 
        'surface_pressure_hpa_mean', 'hrmedia', 'hrmax',
        'dewpoint_2m_c_mean', 'precipitacion_lag1', 'precipitacion_lag7',
        'mes', 'dia_del_anio', 'estacion_invierno', 'estacion_primavera'
    ]
    
    # Ahora sí las encontrará porque las acabamos de crear
    features_cols = [c for c in features_list if c in df.columns]
    
    return df[features_cols], df['bin_prep']

def train_classifier_model_LR(X_train, X_test, y_train, y_test, model_path):
    print(f"\nTraining Logistic Regression...")
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42))
    ])
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"📊 LR Accuracy: {acc:.4f}")
    
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    return model, y_pred

def train_classifier_model_RF(X_train, X_test, y_train, y_test, model_path):
    print(f"\nTraining Random Forest Classifier...")
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42, n_jobs=-1))
    ])
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"📊 RF Accuracy: {acc:.4f}")
    
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    return model, y_pred

def evaluate_classification(y_true, y_pred, model_name):
    """Calcula métricas para modelos de clasificación"""
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    print(f"\n--- Evaluación: {model_name} ---")
    print(f"✅ Accuracy:  {acc:.4f}")
    print(f"🎯 Precision: {prec:.4f} (Calidad de predicción de lluvia)")
    print(f"📢 Recall:    {rec:.4f} (Capacidad de detectar lluvias reales)")
    print(f"⚖️ F1-Score:  {f1:.4f}")

def train_models(df):
    print("\n--- ENTRENANDO MODELOS DE PRECIPITACIÓN ---")
    
    # Preparar datos (Ahora incluye mes y dia_anio)
    X, y = preparar_datos_lluvia(df)
    y = y.astype(int)
    
    split = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    print(f"Datos entrenamiento: {len(X_train)} | Test: {len(X_test)}")
    print(f"Features usadas: {list(X.columns)}")  # Para verificar

    # Rutas
    path_rf = "src/models/modelo_lluvia_rf.pkl"
    path_lr = "src/models/modelo_lluvia_lr.pkl"

    model_rf, y_pred_rf = train_classifier_model_RF(X_train, X_test, y_train, y_test, path_rf)
    model_lr, y_pred_lr = train_classifier_model_LR(X_train, X_test, y_train, y_test, path_lr)

    evaluate_classification(y_test, y_pred_rf, "Random Forest")
    evaluate_classification(y_test, y_pred_lr, "Logistic Regression")

    print(f"\n✅ Modelos de lluvia actualizados correctamente.")
    return model_rf, y_pred_rf, model_lr, y_pred_lr

if __name__ == "__main__":
    try:
        # Ajustamos ruta para ejecución directa
        df_load = pd.read_csv('src/data/processed/data_weather_final.csv')
        train_models(df_load)
    except FileNotFoundError:
        print("⚠️ No se encontró el dataset.")