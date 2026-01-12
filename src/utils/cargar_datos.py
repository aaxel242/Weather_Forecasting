import os
import pandas as pd
import streamlit as st

def cargar_datos():
    # Carga datos de CSV definitivo (data_weather_final.csv) desde carpeta processed.
    # Permite concatenar datos adicionales si el usuario sube archivos. Retorna DataFrame.
    data = None

    # CSV definitivo (base)
    ruta_csv_default = os.path.join("src", "data", "processed", "data_weather_final.csv")
    if os.path.exists(ruta_csv_default):
        data = pd.read_csv(ruta_csv_default)
    else:
        st.warning("⚠️ No se encontró el CSV definitivo. Compila los datos primero.")
        data = pd.DataFrame()  # vacío para poder concatenar si hay uploader

    return data
    