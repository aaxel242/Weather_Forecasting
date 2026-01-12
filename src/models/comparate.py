import pandas as pd

from src.utils.limpieza import limpiar_datos
from src.utils.imputar_datos import imputar_datos
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import (confusion_matrix, accuracy_score, 
                             precision_score, recall_score, f1_score)

df = pd.read_csv('src/data/processed/data_weather_final.csv')

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

# LIMPIEZA CONTROLADA DE FEATURES TEMPORALES
df = df.dropna(subset=['pressure_delta'])

FEATURES = [
    'precipitacion_lag1', 
    'rain_yesterday_bin',  # Indicador binario de lluvia el día anterior.
    'pressure_delta',  # Cambio de presión respecto al día anterior.
    'surface_pressure_hpa_mean', 
    'cloudcover__mean',
    'cloudcover__max',
    'hrmedia', # Humedad relativa media diaria.
    'hrmax', # Máxima humedad del día
    'dewpoint_2m_c_mean',  # Punto de rocío medio a 2 metros.
    'mes', 
    'dia_del_anio', 
    'estacion_invierno', 
    'estacion_verano', 
    'estacion_otoo',
    'estacion_primavera'
]

def imprimir_metricas(y_test, y_pred, nombre_modelo):
    # Calcula e imprime métricas de clasificación: Accuracy, Precisión, Recall, F1 y Matriz de Confusión.
    # Parámetros: y_test (valores reales), y_pred (predicciones), nombre_modelo (etiqueta a mostrar).
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

def logistic_cv_smote_grind(df):
    # Entrena Regresión Logística con GridSearch y SMOTE para desbalance de clases.
    # Parámetro: df (DataFrame con features y target 'bin_prep'). Retorna modelo entrenado.
    print("\n--- Entrenando Logistic Regression con GridSearch + SMOTE ---")

    df_model = df[FEATURES + ['bin_prep']].copy()

    if df_model.isna().any().any():
        raise ValueError("❌ Hay NaNs después de la imputación (Logistic).")

    X = df_model[FEATURES]
    y = df_model['bin_prep']

    # X_train, X_test, y_train, y_test = temporal_train_test_split(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False, random_state=42)

    pipeline_lr = Pipeline([
        ('scaler', StandardScaler()),
        ('smote', SMOTE(random_state=42)),
        ('lr', LogisticRegression(
            max_iter=1000,
            random_state=42
        ))
    ])

    param_grid_lr = {
        'lr__C': [0.01, 0.1, 1, 10],
        'lr__solver': ['lbfgs']
    }

    grid_lr = GridSearchCV(
        pipeline_lr,
        param_grid_lr,
        cv=5,
        scoring='f1',
        n_jobs=-1
    )

    grid_lr.fit(X_train, y_train)

    best_model = grid_lr.best_estimator_

    y_pred = best_model.predict(X_test)

    imprimir_metricas(y_test, y_pred, "LOGISTIC REGRESSION")

    return best_model

def randomforest_cv_smote_grind(df):
    # Entrena RandomForest con GridSearch y SMOTE para desbalance de clases.
    # Parámetro: df (DataFrame con features y target). Retorna pipeline entrenado.
    print("\n--- Entrenando RandomForest con GridSearch ---")

    df_model = df[FEATURES + ['bin_prep']].copy()

    if df_model.isna().any().any():
        raise ValueError("❌ Hay NaNs después de la imputación (RF).")

    X = df_model[FEATURES]
    y = df_model['bin_prep']
    
    # X_train, X_test, y_train, y_test = temporal_train_test_split(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False, random_state=42)

    pipeline_rf = Pipeline([
        ('smote', SMOTE(random_state=42)),
        ('rf', RandomForestClassifier(
            random_state=42,
            # class_weight='balanced',
            # n_estimators=500,
            # max_depth=8,
            # min_samples_leaf=10,
            # min_samples_split=5
        ))
    ])

    # Dejamos que GridSearch busque el equilibrio usando F1
    param_grid = {
        'rf__n_estimators': [500],
        'rf__max_depth': [8],
        'rf__min_samples_leaf': [10]
    }

    grid_rf = GridSearchCV(pipeline_rf, param_grid, cv=5, scoring='f1', n_jobs=-1)
    grid_rf.fit(X_train, y_train)

    y_pred = grid_rf.best_estimator_.predict(X_test)

    imprimir_metricas(y_test, y_pred, "RANDOM FOREST")

    return pipeline_rf

def smv_cv_smote_grind(df):
    # Entrena SVM con GridSearch y SMOTE para clasificación binaria de lluvia.
    # Parámetro: df (DataFrame con features y target). Retorna modelo entrenado con escalado.
    print("\n--- Entrenando SVM con GridSearch + SMOTE ---")

    df_model = df[FEATURES + ['bin_prep']].copy()

    if df_model.isna().any().any():
        raise ValueError("❌ Hay NaNs después de la imputación (SVM).")

    X = df_model[FEATURES]
    y = df_model['bin_prep']

    # X_train, X_test, y_train, y_test = temporal_train_test_split(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False, random_state=42)

    pipeline_svm = Pipeline([
        ('scaler', StandardScaler()),
        ('smote', SMOTE(random_state=42)),
        ('svc', SVC(
            probability=True,
            random_state=42,
        ))
    ])

    param_grid_svm = {
        'svc__C': [0.1, 1, 10],
        'svc__kernel': ['rbf'],
        'svc__gamma': ['scale']
    }

    grid_svm = GridSearchCV(
        pipeline_svm,
        param_grid_svm,
        cv=5,
        scoring='f1',
        n_jobs=-1
    )

    grid_svm.fit(X_train, y_train)

    best_model = grid_svm.best_estimator_

    y_pred = best_model.predict(X_test)

    imprimir_metricas(y_test, y_pred, "SVM")

    return best_model

def temporal_train_test_split(X, y, test_size=0.2):
    split_idx = int(len(X) * (1 - test_size))

    X_train = X.iloc[:split_idx]
    X_test  = X.iloc[split_idx:]
    y_train = y.iloc[:split_idx]
    y_test  = y.iloc[split_idx:]

    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    try:
        # Ejecutar ambos modelos
        best_rf = randomforest_cv_smote_grind(df)
        best_svm = smv_cv_smote_grind(df)
        best_lr  = logistic_cv_smote_grind(df)
        print("\n✅ Proceso completado con éxito.")
        
    except FileNotFoundError:
        print("❌ Error: No se encontró el archivo CSV.")
