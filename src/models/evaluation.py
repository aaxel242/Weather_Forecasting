from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_absolute_error, r2_score, mean_squared_error
import numpy as np
import streamlit as st

def evaluate_precipitation(y_true, y_pred, nombre):
    st.markdown(f"#### {nombre}")
    
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # Creamos 4 columnas para las métricas de clasificación
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy", f"{acc:.1%}")
    m2.metric("Precisión", f"{prec:.1%}")
    m3.metric("Recall", f"{rec:.1%}")
    m4.metric("F1-Score", f"{f1:.2f}")

def evaluate_temperature(y_true, y_pred, nombre):
    st.markdown(f"#### {nombre}")
    
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    
    col1, col2, col3, col4 = st.columns(4)

    # Delta indica si el error es aceptable (verde si es bajo)
    col1.metric("Error Medio absoluto (MAE)", f"{mae:.2f} °C")
    col2.metric("Error Cuadrático (MSE)", f"{mse:.2f}")
    col3.metric("Raíz Error Cuad. (RMSE)", f"{rmse:.2f} °C")
    col4.metric("Coeficiente R²", f"{r2:.2f}", help="Indica qué tan bien el modelo sigue la tendencia. 1.0 es perfecto.")
    
    if mae < 2.0:
        st.success(f"El {nombre} tiene un error muy bajo (menor a 2°C).")
    elif mae > 4.0:
        st.warning(f"El {nombre} presenta desviaciones importantes.")