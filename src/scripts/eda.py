import streamlit as st
import pandas as pd
import joblib
import os
# Importaciones de tus módulos
from src.utils.cargar_datos import cargar_datos
from src.utils.limpieza import limpiar_datos
from src.utils.imputar_datos import imputar_datos
from src.utils.p_value import correlation_heatmap
from src.utils.data_analysis import basic_stats
from src.utils.visualize_data import visualization_Data
from src.utils.show_evaluation import show_evaluation

st.set_page_config(page_title="Weather Forecasting", layout="wide")

def render_eda_section():
    """
    Orquesta sección de análisis exploratorio de datos (EDA) con 4 pestañas.
    Carga, limpia, imputa datos y muestra estadísticas, correlaciones, visualizaciones y modelos.
    Parámetros: ninguno. Retorna: None (renderiza en Streamlit).
    """

    data = cargar_datos()
    
    if data is not None:
        data_clean = limpiar_datos(data)

        if data_clean.isna().values.any():
            data_final = imputar_datos(data_clean)
        else:
            data_final = data_clean

        tab1, tab2, tab3, tab4 = st.tabs(["Estadísticas", "Correlaciones", "Visualización", "Resultados de los modelos"])
        with tab1:
            basic_stats(data_final)
        with tab2:          
            correlation_heatmap(data_final)
        with tab3:
            visualization_Data(data_final)
        with tab4:
            show_evaluation(data_final)