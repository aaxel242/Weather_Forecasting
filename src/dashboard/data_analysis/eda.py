import streamlit as st
import pandas as pd
import joblib
import os
# Importaciones de tus módulos
from src.utils.cargar_datos import cargar_datos
from src.utils.limpieza import limpiar_datos
from src.utils.imputar_datos import imputar_datos
from src.utils.p_value_2 import correlation_heatmap
from src.dashboard.data_analysis.data_analysis import basic_stats
from src.dashboard.data_analysis.visualize_data import visualization_Data
from src.dashboard.data_analysis.show_evaluation import show_evaluation

st.set_page_config(page_title="Weather Forecasting AI", layout="wide")

def main():
    st.title("Weather Forecasting - Data Analysis")

    data = cargar_datos()
    
    if data is not None:
        if data.isna().values.any():
            data_clean = limpiar_datos(data)
            data_imput = imputar_datos(data_clean)
        else:
            data_imput = limpiar_datos(data)

        tab1, tab2, tab3 = st.tabs(["Estadísticas", "Correlaciones", "Visualización"])
        with tab1:
            basic_stats(data_imput)
            show_evaluation(data_imput)
        with tab2:          
            correlation_heatmap(data_imput)
        with tab3:
            visualization_Data(data_imput)

if __name__ == "__main__":
    main()