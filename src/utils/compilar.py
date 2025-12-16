import os
import pandas as pd
import streamlit as st
from utils.unir_json import unir_json_a_csv

def compilar_datos():
    """
    Compila todos los ficheros JSON de la carpeta en un único CSV.
    """
    carpeta_json = "src/datos/json"
    salida_csv = "src/datos/csv/datos_clima_definitivo.csv"

    df_final = unir_json_a_csv(carpeta_json, salida_csv)

    if df_final is not None:
        st.sidebar.success("✅ Datos compilados correctamente")
    else:
        st.sidebar.error("⚠️ No se pudieron compilar los datos")

    return df_final