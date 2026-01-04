import os
import streamlit as st
import joblib
import pandas as pd
from src.models.evaluation import evaluate_precipitation, evaluate_temperature

def show_evaluation(data):
    print("Evaluación de Modelos Pre-entrenados")

    modelos = { 
        "Modelo General de Lluvia": "src/models/modelo_lluvia.pkl", 
        "Modelo de Temperatura Máxima": "src/models/modelo_tmax.pkl", 
        "Modelo de Temperatura Mínima": "src/models/modelo_tmin.pkl" 
    }

    # Preparamos features y targets
    leaky = ["bin_prep", "tmax", "tmin", "date", "fecha"]
    X = data.drop(columns=[c for c in leaky if c in data.columns], errors="ignore")

    y_lluvia = data["bin_prep"].astype(int) if "bin_prep" in data else None
    y_tmax = data["tmax"] if "tmax" in data else None
    y_tmin = data["tmin"] if "tmin" in data else None

    # Split 80/20
    split = int(len(X) * 0.8)
    X_test = X.iloc[split:]

    y_test_lluvia = y_lluvia.iloc[split:] if y_lluvia is not None else None
    y_test_tmax = y_tmax.iloc[split:] if y_tmax is not None else None
    y_test_tmin = y_tmin.iloc[split:] if y_tmin is not None else None

    # Evaluación por modelo
    for nombre, ruta in modelos.items():

        if not os.path.exists(ruta):
            st.warning(f"⚠️ {nombre} no encontrado en la ruta: {ruta}")
            continue

        try:
            modelo = joblib.load(ruta)
            print(f"✅ {nombre} cargado correctamente.")
            print(modelo)

            # -----------------------------
            # CLASIFICACIÓN (lluvia)
            # -----------------------------
            if "Lluvia" in nombre:
                y_pred = modelo.predict(X_test)

                # evaluate_classification necesita 3 argumentos
                # duplicamos y_pred para cumplir la firma
                evaluate_precipitation(y_test_lluvia, y_pred)

            # -----------------------------
            # REGRESIÓN (tmax / tmin)
            # -----------------------------
            else:
                y_true = y_test_tmax if "Máxima" in nombre else y_test_tmin
                y_pred = modelo.predict(X_test)

                evaluate_temperature(y_true, y_pred)

        except Exception as e:
            st.error(f"❌ Error al cargar {nombre}: {e}")


# Cargar dataset correctamente
df = pd.read_csv("src/data/processed/data_weather.csv")
show_evaluation(df)