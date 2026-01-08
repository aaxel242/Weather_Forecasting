import os
import streamlit as st
import joblib
import pandas as pd
import numpy as np
from src.models.evaluation import evaluate_precipitation, evaluate_temperature

def show_evaluation(data):
    st.header("Evaluación de Modelos Pre-entrenados")
    
    modelos_config = [
        {"nombre": "Modelo de Lluvia", "ruta": "src/models/modelo_lluvia.pkl", "tipo": "clasificacion"},
        {"nombre": "Temperatura Máxima", "ruta": "src/models/modelo_tmax.pkl", "tipo": "regresion"},
        {"nombre": "Temperatura Mínima", "ruta": "src/models/modelo_tmin.pkl", "tipo": "regresion"}
    ]

    for config in modelos_config:
        with st.expander(f"Detalles: {config['nombre']}", expanded=True):
            if not os.path.exists(config["ruta"]):
                st.warning(f"No encontrado: {config['ruta']}")
                continue

            try:
                modelo = joblib.load(config["ruta"])
                # Trabajamos sobre una copia limpia para cada modelo
                df_temp = data.copy()

                # --- 1. PREPARACIÓN DE COLUMNAS (Mapeo del CSV al Modelo) ---
                if 'date' in df_temp.columns:
                    df_temp['date'] = pd.to_datetime(df_temp['date'])
                    df_temp = df_temp.sort_values('date')

                # Mapeos para LLUVIA
                if 'precipitacion_lag1' in df_temp.columns:
                    df_temp['prec_yesterday'] = df_temp['precipitacion_lag1']
                    df_temp['rain_yesterday_bin'] = (df_temp['prec_yesterday'] > 0.1).astype(int)
                
                if 'dia_del_anio' in df_temp.columns:
                    df_temp['dia_anio'] = df_temp['dia_del_anio']

                if 'surface_pressure_hpa_mean' in df_temp.columns:
                    # Calculamos el delta de presión (diferencia con el día anterior)
                    df_temp['pressure_delta'] = df_temp['surface_pressure_hpa_mean'].diff()

                # Mapeos para TEMPERATURA
                if 'temp_max_lag1' in df_temp.columns:
                    df_temp['tmax_yesterday'] = df_temp['temp_max_lag1']
                if 'temp_min_lag1' in df_temp.columns:
                    df_temp['tmin_yesterday'] = df_temp['temp_min_lag1']

                # --- 2. IDENTIFICAR COLUMNAS REQUERIDAS ---
                if hasattr(modelo, 'feature_names_in_'):
                    features_modelo = list(modelo.feature_names_in_)
                else:
                    features_modelo = list(modelo.steps[-1][1].feature_names_in_)

                # Definir Target según el modelo
                if config["tipo"] == "clasificacion":
                    target_col = "bin_prep"
                else:
                    target_col = "tmax" if "Máxima" in config["nombre"] else "tmin"

                # --- 3. LIMPIEZA DE FILAS (Sincronización de X e y) ---
                # Eliminamos filas donde falte alguna feature o el target
                columnas_necesarias = features_modelo + [target_col]
                df_clean = df_temp.dropna(subset=columnas_necesarias)

                if df_clean.empty:
                    st.error(f"No hay datos suficientes para evaluar {config['nombre']} tras limpiar NaNs.")
                    continue

                # --- 4. DIVISIÓN Y PREDICCIÓN ---
                # Ahora X e y salen del MISMO dataframe limpio
                X = df_clean[features_modelo]
                y = df_clean[target_col]

                split_idx = int(len(X) * 0.8)
                X_test = X.iloc[split_idx:]
                y_true = y.iloc[split_idx:]

                if config["tipo"] == "clasificacion":
                    # Aplicamos el umbral de 0.26 para lluvia
                    y_probs = modelo.predict_proba(X_test)[:, 1]
                    y_pred = (y_probs >= 0.26).astype(int)
                    evaluate_precipitation(y_true, y_pred, config["nombre"])
                else:
                    y_pred = modelo.predict(X_test)
                    evaluate_temperature(y_true, y_pred, config["nombre"])

            except Exception as e:
                st.error(f"❌ Error en {config['nombre']}: {e}")