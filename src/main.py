import joblib
import streamlit as st
import os
import pandas as pd
from PIL import Image

from utils.cargar_datos import cargar_datos
from utils.limpieza import limpiar_datos
from utils.imputar_datos import imputar_datos
from scripts.eda import eda_interactivo
from models.prediccion import predict_with_model
from models.train_model import train_classifier_model_LR, train_classifier_model_RF
from models.evaluation import evaluate_classification

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
            
            # 2. CONVERSIÓN CRÍTICA A ENTERO
            labels = data_imput[target_column_prep].astype(float).astype(int)
            # SELECCIÓN DE MÉTODO
            model_path = f"src/models/"
            model_path_rf = f"{model_path}/model_lluvia_rf.pkl"
            model_path_lr = f"{model_path}/model_lluvia_lr.pkl"

            # BOTÓN ENTRENAR
            
            train_classifier_model_RF(features, labels, model_path=model_path_rf)
            train_classifier_model_LR(features, labels, model_path=model_path_lr)

            # BOTÓN EVALUAR
            model_rf, y_pred_rf = train_classifier_model_RF(features, labels, model_path=model_path_rf)
            model_lr, y_pred_lr = train_classifier_model_LR(features, labels, model_path=model_path_lr)
            y_true = labels.iloc[int(len(labels) * 0.8):]
            metrics_rf = evaluate_classification(y_true, y_pred_rf)
            metrics_lr = evaluate_classification(y_true, y_pred_lr)
            
            predict_with_model(model_path_lr, model_path_rf, features, rango)

if __name__ == "__main__":
    main()