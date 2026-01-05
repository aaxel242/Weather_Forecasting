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
                
                # --- OBTENER LAS COLUMNAS QUE EL MODELO REALMENTE ESPERA ---
                # Esto extrae los nombres de las columnas usados durante el entrenamiento
                if hasattr(modelo, 'feature_names_in_'):
                    features_esperadas = modelo.feature_names_in_
                else:
                    # Si usaste Pipeline con StandardScaler, a veces está en el clasificador
                    features_esperadas = modelo.steps[-1][1].feature_names_in_

                # --- PREPARAR EL DATAFRAME TEMPORAL ---
                X_eval = data.copy()

                # 1. Corregir nombres (dia_del_anio -> dia_anio)
                if 'dia_del_anio' in X_eval.columns:
                    X_eval['dia_anio'] = X_eval['dia_del_anio']
                
                # 2. Crear tmax_yesterday y tmin_yesterday (Parece que tus modelos las necesitan)
                if 'temp_max_lag1' in X_eval.columns:
                    X_eval['tmax_yesterday'] = X_eval['temp_max_lag1']
                if 'temp_min_lag1' in X_eval.columns:
                    X_eval['tmin_yesterday'] = X_eval['temp_min_lag1']

                # 3. Seleccionar SOLO las columnas que el modelo conoce y en el ORDEN correcto
                # Esto soluciona el error de "Feature names unseen" y "missing"
                try:
                    X_test_final = X_eval[features_esperadas]
                except KeyError as e:
                    st.error(f"Faltan columnas críticas para este modelo: {e}")
                    continue

                # --- PREDICCIÓN ---
                split = int(len(X_test_final) * 0.8)
                X_test_input = X_test_final.iloc[split:]
                y_pred = modelo.predict(X_test_input)

                # --- EVALUACIÓN (Igual que antes) ---
                if config["tipo"] == "clasificacion":
                    y_true = data["bin_prep"].iloc[split:].astype(int)
                    evaluate_precipitation(y_true, y_pred, config["nombre"])
                else:
                    target_col = "tmax" if "Máxima" in config["nombre"] else "tmin"
                    y_true = data[target_col].iloc[split:]
                    evaluate_temperature(y_true, y_pred, config["nombre"])

            except Exception as e:
                st.error(f"❌ Error en {config['nombre']}: {e}")