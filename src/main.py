import streamlit as st
import os
import pandas as pd
from PIL import Image

from utils.cargar_datos import cargar_datos
from utils.limpieza import limpiar_datos
from utils.imputar_datos import imputar_datos
from scripts.eda import eda_interactivo
from models.prediccion import load_model, predict_with_model
from models.train_model import train_regression_model 
from models.evaluation import evaluate_regression

def main():
    # Imagen de inicio
    image_path = os.path.join("src", "images", "imagen_entrada.png")
    if os.path.exists(image_path):
        st.image(Image.open(image_path))

    st.sidebar.header("📅 Opciones de entrada")
    fecha = st.sidebar.date_input("Selecciona la fecha")
    rango = st.sidebar.slider("Rango de días a predecir", 1, 7, 3)

    # Cargar y procesar datos
    data = cargar_datos() 
    data_imput = None

    if data is not None:
        # PASO 1: Limpiar comillas y normalizar (SIN borrar filas)
        data_clean = limpiar_datos(data)        
        # PASO 2: Rellenar huecos con tu lógica de imputación
        data_imput = imputar_datos(data_clean)  
               
        eda_interactivo(data_imput)
        
        # Exportación
        st.sidebar.markdown("### 📁 Exportar Datos")
        if st.sidebar.button("💾 Guardar en Servidor"):
            ruta_dir = os.path.join("src", "datos", "processed")
            os.makedirs(ruta_dir, exist_ok=True)
            data_imput.to_csv(os.path.join(ruta_dir, "datos_clima_limpios.csv"), index=True)
            st.sidebar.success("✅ Guardado en servidor")

        csv_buffer = data_imput.to_csv(index=True).encode('utf-8')
        st.sidebar.download_button(
            label="📥 Descargar CSV limpio e imputado",
            data=csv_buffer,
            file_name='data_weather_oficial.csv',
            mime='text/csv',
        )
            
    st.sidebar.markdown("---")
    target_column = "tmed" 

    if data_imput is not None and target_column in data_imput.columns:
        features = data_imput.drop(columns=[target_column], errors='ignore')
        labels = data_imput[target_column]

        if st.sidebar.button("🏋🧠 Entrenar modelo"):
            train_regression_model(features, labels, model_path="src/models/trained_regression_tmed_model.pkl")
            st.sidebar.success(f"Modelo entrenado para '{target_column}'.")

        if st.sidebar.button("🔍📊 Evaluar rendimiento"):
            model, y_pred_test = train_regression_model(features, labels, model_path="src/models/trained_regression_tmed_model.pkl")
            y_true_test = labels.iloc[int(len(labels) * 0.8):]
            st.sidebar.json(evaluate_regression(y_true_test, y_pred_test))

        if st.button("Generar predicción"):
            model = load_model("src/models/trained_regression_tmed_model.pkl")
            st.write("Predicción generada:", predict_with_model(model, features))

if __name__ == "__main__":
    main()