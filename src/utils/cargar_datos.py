import os
import pandas as pd
import streamlit as st

def cargar_datos():
    """
    Carga datos para el pipeline:
    - Siempre usa el CSV definitivo generado a partir de los JSON base.
    - Si el usuario sube datos adicionales, se concatenan a la base.
    """
    data = None

    # CSV definitivo (base)
    ruta_csv_default = os.path.join("src", "data", "processed", "data_weather_final.csv")
    if os.path.exists(ruta_csv_default):
        data = pd.read_csv(ruta_csv_default)
    else:
        st.warning("⚠️ No se encontró el CSV definitivo. Compila los datos primero.")
        data = pd.DataFrame()  # vacío para poder concatenar si hay uploader

    return data