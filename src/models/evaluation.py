import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score
)

def evaluate_classification(y_true, y_pred_rf, y_pred_lr):
    # Forzamos conversión a entero para evitar el error de '0.0' vs 0
    y_true = np.array(y_true).astype(int)
    y_pred_rf = np.array(y_pred_rf).astype(int)
    y_pred_lr = np.array(y_pred_lr).astype(int)

    st.write("### Evaluación de Modelos de Clasificación")
    st.json({
        "Precisión Global (Accuracy)": f"{accuracy_score(y_true, y_pred_rf):.2%}",
        "Precisión (Evitar Falsos Positivos)": f"{precision_score(y_true, y_pred_rf, zero_division=0):.2%}",
        "Recall (Detectar Lluvia Real)": f"{recall_score(y_true, y_pred_rf, zero_division=0):.2%}",
    })

    st.write("#### Modelo de Regresión Logística")
    st.json({
        "Precisión Global (Accuracy)": f"{accuracy_score(y_true, y_pred_lr):.2%}",
        "Precisión (Evitar Falsos Positivos)": f"{precision_score(y_true, y_pred_lr, zero_division=0):.2%}",
        "Recall (Detectar Lluvia Real)": f"{recall_score(y_true, y_pred_lr, zero_division=0):.2%}",
    })