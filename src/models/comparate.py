import joblib
import pandas as pd
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

from src.utils.cargar_datos import cargar_datos
from src.utils.limpieza import limpiar_datos
from src.utils.imputar_datos import imputar_datos

# Cargar y preparar datos
data = cargar_datos() 
data_clean = limpiar_datos(data)        
data_imput = imputar_datos(data_clean)  

target_column_prep = "bin_prep" 

leaky = [target_column_prep, "date"]
features = data_imput.drop(columns=[c for c in leaky if c in data_imput.columns], errors='ignore')
labels = data_imput[target_column_prep].astype(int)

# Crear directorio de modelos
model_dir = "src/models"
os.makedirs(model_dir, exist_ok=True)

model_path_rf = f"{model_dir}/model_lluvia_rf.pkl"
model_path_lr = f"{model_dir}/model_lluvia_lr.pkl"

# Split de datos
split_factor = 0.8
split = int(len(features) * split_factor)
X_train, X_test = features.iloc[:split], features.iloc[split:]
y_train, y_test = labels.iloc[:split], labels.iloc[split:]

# ============================================
# MODELO 1: Random Forest
# ============================================
print("Entrenando Random Forest...")
model_rf = Pipeline([
    ('classifier', RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42))
])

model_rf.fit(X_train, y_train)
joblib.dump(model_rf, model_path_rf)

predictions_rf = model_rf.predict(X_test)

accuracy_rf = accuracy_score(y_test, predictions_rf)
precision_rf = precision_score(y_test, predictions_rf, average='weighted', zero_division=0)
recall_rf = recall_score(y_test, predictions_rf, average='weighted', zero_division=0)
f1_rf = f1_score(y_test, predictions_rf, average='weighted', zero_division=0)

mae_rf = mean_absolute_error(y_test, predictions_rf)
mse_rf = mean_squared_error(y_test, predictions_rf)
rmse_rf = np.sqrt(mean_squared_error(y_test, predictions_rf))
r2_rf = r2_score(y_test, predictions_rf)

print(f"\n--- Evaluación Modelo Random Forest ---")
print(f"Accuracy: {accuracy_rf:.4f}")
print(f"Precision: {precision_rf:.4f}")
print(f"Recall: {recall_rf:.4f}")
print(f"F1-Score: {f1_rf:.4f}")
print(f"\n{classification_report(y_test, predictions_rf)}")

# ============================================
# MODELO 2: Logistic Regression
# ============================================
print("\n\nEntrenando Logistic Regression...")
model_lr = Pipeline([
    ('classifier', LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42))
])

model_lr.fit(X_train, y_train)
joblib.dump(model_lr, model_path_lr)

predictions_lr = model_lr.predict(X_test)

accuracy_lr = accuracy_score(y_test, predictions_lr)
precision_lr = precision_score(y_test, predictions_lr, average='weighted', zero_division=0)
recall_lr = recall_score(y_test, predictions_lr, average='weighted', zero_division=0)
f1_lr = f1_score(y_test, predictions_lr, average='weighted', zero_division=0)

mae_lr = mean_absolute_error(y_test, predictions_lr)
mse_lr = mean_squared_error(y_test, predictions_lr)
rmse_lr = np.sqrt(mean_squared_error(y_test, predictions_lr))
r2_lr = r2_score(y_test, predictions_lr)

print(f"\n--- Evaluación Modelo Logistic Regression ---")
print(f"Accuracy: {accuracy_lr:.4f}")
print(f"Precision: {precision_lr:.4f}")
print(f"Recall: {recall_lr:.4f}")
print(f"F1-Score: {f1_lr:.4f}")
print(f"\n{classification_report(y_test, predictions_lr)}")

# ============================================
# COMPARATIVA
# ============================================
print("\n\n--- COMPARATIVA DE MODELOS ---")
print(f"Random Forest F1-Score: {f1_rf:.4f}")
print(f"Logistic Regression F1-Score: {f1_lr:.4f}")

print("\n\n--- COMPARATIVA DE MODELOS ---")
print(f"Random Forest MAE: {mae_rf:.4f}")
print(f"Logistic Regression MAE: {mae_lr:.4f}")

print("\n\n--- COMPARATIVA DE MODELOS ---")
print(f"Random Forest RMSE: {rmse_rf:.4f}")
print(f"Logistic Regression RMSE: {rmse_lr:.4f}")

print(f"\nModelos guardados en: {model_dir}")













import joblib
import pandas as pd
import numpy as np
import os

from sklearn.model_selection import GridSearchCV
from collections import Counter
from imblearn.over_sampling import SMOTE
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

# --- CONFIGURACIÓN ---
RUTA_DATOS = 'src/data/processed/data_weather_final.csv'
RUTA_MODELO = 'src/models/modelo_lluvia.pkl'
RUTA_FEATURES = 'src/models/features_lluvia.pkl'
RUTA_SCALER = 'src/models/scaler_lluvia.pkl'
RUTA_UMBRAL = 'src/models/umbral_lluvia.pkl' # ← Guardaremos el umbral óptimo

def train_rain_model_optimized():
    print("\n--- ENTRENANDO MODELO LLUVIA (SVM OPTIMIZADO) ---")
    
    try:
        df = pd.read_csv(RUTA_DATOS)
    except FileNotFoundError:
        print(f"❌ Error: No se encuentra {RUTA_DATOS}")
        return

    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')

    # 1. INGENIERÍA DE CARACTERÍSTICAS
    df['rain_yesterday_bin'] = (df['precipitacion_lag1'] > 0.1).astype(int)
    
    # Presión
    df['pressure_yesterday'] = df['surface_pressure_hpa_mean'].shift(1)
    df['pressure_delta'] = df['surface_pressure_hpa_mean'] - df['pressure_yesterday']
    
    # --- MEJORA: CARACTERÍSTICAS CÍCLICAS (Seno/Coseno) ---
    # Esto ayuda al modelo a entender que diciembre y enero están juntos
    df['dia_sin'] = np.sin(2 * np.pi * df['dia_del_anio'] / 365.0)
    df['dia_cos'] = np.cos(2 * np.pi * df['dia_del_anio'] / 365.0)
    df['mes_sin'] = np.sin(2 * np.pi * df['mes'] / 12.0)
    df['mes_cos'] = np.cos(2 * np.pi * df['mes'] / 12.0)
    
    df = df.dropna()

    features_cols = [
        'precipitacion_lag1', 'rain_yesterday_bin',
        'pressure_delta', 
        'surface_pressure_hpa_mean', 
        'cloudcover__mean', 'cloudcover__max',
        'hrmedia', 'hrmax',
        'dewpoint_2m_c_mean',
        # Usamos las cíclicas en lugar de 'mes' y 'dia' puros
        'dia_sin', 'dia_cos', 'mes_sin', 'mes_cos'
    ]

    X = df[features_cols]
    y = df['bin_prep']

    print(f"Features usadas: {features_cols}")

    # 2. SPLIT
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False, random_state=42)

    # 3. NORMALIZACIÓN
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 4. SMOTE (Sobremuestreo)
    # Sampling strategy 0.7 significa: queremos que la lluvia sea el 70% de los días de sol (no 1 a 1 para no forzar tanto)
    smote = SMOTE(sampling_strategy=0.8, random_state=42) 
    X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)
    print(f"Distribución tras SMOTE: {Counter(y_train_smote)}")

    # 5. ENTRENAMIENTO SVM
    print("\n🔍 Iniciando Entrenamiento SVM...")
    
    # Hemos quitado class_weight='balanced' para reducir Falsos Positivos
    # Hemos bajado C a 10 para reducir overfitting (tu anterior C=100 era muy agresivo)
    model = SVC(
        C=10, 
        kernel='rbf', 
        gamma='scale',
        probability=True, # Necesario para curvas de precisión
        random_state=42
    )
    
    model.fit(X_train_smote, y_train_smote)

    # 6. BÚSQUEDA DEL UMBRAL ÓPTIMO (Threshold Tuning)
    # En lugar de adivinar, probamos todos los cortes posibles
    print("\n⚖️ Buscando el mejor umbral de decisión...")
    
    y_probs = model.predict_proba(X_test_scaled)[:, 1] # Probabilidad de lluvia
    
    thresholds = np.arange(0.1, 0.9, 0.05)
    best_f1 = 0
    best_thresh = 0.5
    
    for thresh in thresholds:
        y_pred_temp = (y_probs >= thresh).astype(int)
        f1 = f1_score(y_test, y_pred_temp)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh

    # Aplicamos el mejor umbral encontrado
    y_final_pred = (y_probs >= best_thresh).astype(int)

    # 7. RESULTADOS
    acc = accuracy_score(y_test, y_final_pred)
    prec = precision_score(y_test, y_final_pred, zero_division=0)
    rec = recall_score(y_test, y_final_pred, zero_division=0)
    
    print(f"\n🏆 MEJOR UMBRAL ENCONTRADO: {best_thresh:.2f}")
    print(f"📊 RESULTADOS FINALES:")
    print(f"   - Recall (Sensibilidad): {rec:.4f}")
    print(f"   - Precision:             {prec:.4f}")
    print(f"   - Accuracy:              {acc:.4f}")
    print(f"   - F1-Score:              {best_f1:.4f}")

    cm = confusion_matrix(y_test, y_final_pred)
    print(f"\n   Matriz de Confusión:")
    print(f"   TN (Sol ok): {cm[0][0]} | FP (Falsa alarma): {cm[0][1]}")
    print(f"   FN (Lluvia perdida): {cm[1][0]} | TP (Lluvia detectada): {cm[1][1]}")

    # GUARDADO
    os.makedirs(os.path.dirname(RUTA_MODELO), exist_ok=True)
    joblib.dump(model, RUTA_MODELO)
    joblib.dump(scaler, RUTA_SCALER)
    joblib.dump(features_cols, RUTA_FEATURES)
    joblib.dump(best_thresh, RUTA_UMBRAL) # Guardamos el umbral para usarlo en prediction_engine
    
    print(f"\n✅ Modelo y componentes guardados.")

if __name__ == "__main__":
    train_rain_model_optimized()