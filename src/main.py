import streamlit as st
import os
import pandas as pd
from PIL import Image

# Importamos tu lógica de lags y modelos
from data.add_lags import add_lag_features 
from utils.cargar_datos import cargar_datos
from utils.limpieza import limpiar_datos
from utils.imputar_datos import imputar_datos

from models.train_model import train_models
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

        # 1. CARGA Y LIMPIEZA INICIAL
        data = cargar_datos() 
        if data is None:
            st.error("No se pudieron cargar los datos.")
            return

        data_clean = limpiar_datos(data)        
        data_imput = imputar_datos(data_clean)  
        
        # 2. GENERAR TARGETS (LAGS)
        # IMPORTANTE: add_lag_features debe devolver el dataframe procesado
        # Si tu archivo add_lags.py lee de disco, asegúrate de que guarde y cargue correctamente
        st.info("Generando variables de predicción futura...")
        data_final = add_lag_features(data_imput) 

        # --- LLUVIA  ---
        st.header("☔ Predicción de Lluvia")
        target_rain = "bin_prep" 
        
        if target_rain in data_final.columns:
            leaky_rain = [target_rain, "date", "temp_min_target", "precipitacion_target"]
            features_rain = data_final.drop(columns=[c for c in leaky_rain if c in data_final.columns], errors='ignore')
            labels_rain = data_final[target_rain].astype(int)

            model_path_rf = "src/models/model_lluvia_rf.pkl"
            model_path_lr = "src/models/model_lluvia_lr.pkl"

            # Entrenamiento lluvia
            model_rf, y_pred_rf, model_lr, y_pred_lr = train_models(features_rain, labels_rain, model_path_rf, model_path_lr)
            
            y_true_rain = labels_rain.iloc[int(len(labels_rain) * 0.8):]
            evaluate_classification(y_true_rain, y_pred_rf, y_pred_lr)

        # --- TEMPERATURA MÍNIMA ---
        st.header("🌡️ Predicción de Temperatura Mínima")
        target_temp = "temp_min_target"

        if target_temp in data_final.columns:
            leaky_temp = [target_temp, "date", "bin_prep", "precipitacion_target"]
            features_temp = data_final.drop(columns=[c for c in leaky_temp if c in data_final.columns], errors='ignore')
            labels_temp = data_final[target_temp]

            # Entrenamiento de tu modelo
            model_tmin, pred_tmin = train_temp_min_model(features_temp, labels_temp)
            
            # Evaluación de tu modelo
            y_true_tmin = labels_temp.iloc[int(len(labels_temp) * 0.8):]
            evaluate_temperature(y_true_tmin, pred_tmin)

            # --- BLOQUE C: AGENTE AI (CONSEJOS REALES) ---
            st.markdown("---")
            st.subheader("🤖 Agente AI: Tu Asistente Personal")
            
            # USAMOS .iloc[-1] para forzar a Pandas a coger la ÚLTIMA POSICIÓN
            ultima_temp_pred = pred_tmin.iloc[-1] 
            ultima_lluvia_pred = y_pred_rf.iloc[-1] 

            consejo = agente_meteorologico(ultima_temp_pred, ultima_lluvia_pred)

            st.success(f"**Predicción para el próximo periodo:** {ultima_temp_pred:.1f}°C")
            st.info(f"💡 **Recomendación:** {consejo}")

        # 3. MOSTRAR INTERFAZ DE PREDICCIÓN FINAL
        predict_with_model(model_path_lr, model_path_rf, features_rain, rango)

if __name__ == "__main__":
    main()