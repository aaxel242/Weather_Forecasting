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
    ('scaler', StandardScaler()),
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