import streamlit as st
import os
import pandas as pd
from PIL import Image

# Importamos tu lógica de lags y modelos
from data.add_lags import add_lag_features 
from utils.cargar_datos import cargar_datos
from utils.limpieza import limpiar_datos
from utils.imputar_datos import imputar_datos

from models.evaluation import evaluate_classification
from models.prediccion import predict_with_model
from models.train_model_precipitation import train_models
from models.evaluation import evaluate_classification, evaluate_temperature # Importamos ambas
from models.prediccion import predict_with_model, agente_meteorologico     # Importamos tu agente
from models.train_model_temp_min import train_temp_min_model

def main():
    # Imagen de inicio
    image_path = os.path.join("src", "images", "imagen_entrada.png")
    if os.path.exists(image_path):
        st.image(Image.open(image_path))

    rango = st.slider("Rango de días a predecir", 1, 7, 3)
    if st.button("🔄 Cargar y Procesar Datos"):

        # Cargar y procesar datos
        data = cargar_datos() 
        data_imput = None

        if data is not None:
            # PASO 1: Limpiar comillas y normalizar (SIN borrar filas)
            data_clean = limpiar_datos(data)        
            # PASO 2: Rellenar huecos con tu lógica de imputación
            data_imput = imputar_datos(data_clean)  
                
        # DEFINIMOS EL TARGET DE CLASIFICACIÓN
        target_column_prep = "bin_prep"  # 1 si llueve, 0 si no llueve

        if data_imput is not None and target_column_prep in data_imput.columns:
            # 1. Limpieza de columnas "trampa"
            leaky = [target_column_prep, "date"]
            features = data_imput.drop(columns=[c for c in leaky if c in data_imput.columns], errors='ignore')
            
            labels = data_imput[target_column_prep].astype(float).astype(int)

            model_path = f"src/models/"
            model_path_rf = f"{model_path}/model_lluvia_rf.pkl"
            model_path_lr = f"{model_path}/model_lluvia_lr.pkl"

            model_rf, y_pred_rf, model_lr, y_pred_lr = train_models(features, labels, model_path_rf, model_path_lr)

            y_true = labels.iloc[int(len(labels) * 0.8):]
            evaluate_classification(y_true, y_pred_rf, y_pred_lr)
            
            predict_with_model(model_path_lr, model_path_rf, features, rango)
            
if __name__ == "__main__":
    main()