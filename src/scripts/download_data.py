import sys
import os
import matplotlib.pyplot as plt
import streamlit as st
from pathlib import Path 
import pandas as pd

st.title("Show Data Analyst")

# guardamos los datos en la carpeta
original_path = os.path.join(os.getcwd(), "src", "data")

def download_Data() -> pd.DataFrame:
    # descargamos los datos con streamlit

    file_Data = st.file_uploader("Upload your CSV:", type=["csv"])

    if file_Data is not None:

        # cargamos los datos
        data = pd.read_csv(file_Data)

        # verificamos que existe la carpeta de datos
        if not os.path.exists(original_path):
            os.makedirs(original_path)

        # guardamos los datos en la nueva carpeta
        filename = getattr(file_Data, "name", "uploaded_file.csv")
        ruta_completa = os.path.join(original_path, filename)
        data.to_csv(ruta_completa, index=False)

        # lo mostramos
        st.success(f"File '{file_Data.name}' save")
        st.info(f"New Path: {ruta_completa}")

        return data
    