import streamlit as st
import pandas as pd
import joblib
import os
# Importaciones de tus módulos
from src.utils.cargar_datos import cargar_datos
from src.utils.limpieza import limpiar_datos
from src.utils.imputar_datos import imputar_datos
from utils.p_value import correlation_heatmap
from models.data_analysis import basic_stats
from utils.visualize_data import visualization_Data
from utils.show_evaluation import show_evaluation

st.set_page_config(page_title="Weather Forecasting", layout="wide")

def render_eda_section():
    st.title("Weather Forecasting - Data Analysis")

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