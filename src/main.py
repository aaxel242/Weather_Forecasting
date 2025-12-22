import joblib
import streamlit as st
import os
import pandas as pd
from PIL import Image

from utils.cargar_datos import cargar_datos
from utils.limpieza import limpiar_datos
from utils.imputar_datos import imputar_datos
from scripts.eda import eda_interactivo
from models.prediccion import load_model, predict_with_model
from models.train_model import train_classifier_model
from models.evaluation import evaluate_classification

def main():
    # Imagen de inicio
    image_path = os.path.join("src", "images", "imagen_entrada.png")
    if os.path.exists(image_path):
        st.image(Image.open(image_path))

    st.sidebar.header("📅 Opciones de entrada")
    # fecha = st.sidebar.date_input("Selecciona la fecha")
    # rango = st.sidebar.slider("Rango de días a predecir", 1, 7, 3)

    # Cargar y procesar datos
    data = cargar_datos() 
    data_imput = None

    if data is not None:
        # PASO 1: Limpiar comillas y normalizar (SIN borrar filas)
        data_clean = limpiar_datos(data)        
        # PASO 2: Rellenar huecos con tu lógica de imputación
        data_imput = imputar_datos(data_clean)  
               
        eda_interactivo(data_imput)
        
        # # Exportación
        # st.sidebar.markdown("### 📁 Exportar Datos")
        # if st.sidebar.button("💾 Guardar en Servidor"):
        #     ruta_dir = os.path.join("src", "datos", "processed")
        #     os.makedirs(ruta_dir, exist_ok=True)
        #     data_imput.to_csv(os.path.join(ruta_dir, "datos_clima_limpios.csv"), index=True)
        #     st.sidebar.success("✅ Guardado en servidor")

        # csv_buffer = data_imput.to_csv(index=True).encode('utf-8')
        # st.sidebar.download_button(
        #     label="📥 Descargar CSV limpio e imputado",
        #     data=csv_buffer,
        #     file_name='data_weather_oficial.csv',
        #     mime='text/csv',
        # )
            
    st.sidebar.markdown("---")
    # DEFINIMOS EL TARGET DE CLASIFICACIÓN
    target_column = "lluvia_binaria_target" 

    if data_imput is not None and target_column in data_imput.columns:
        # 1. Limpieza de columnas "trampa"
        leaky = [target_column, "precipitacion_target", "temp_max_target", "temp_min_target", "date"]
        features = data_imput.drop(columns=[c for c in leaky if c in data_imput.columns], errors='ignore')
        
        # 2. CONVERSIÓN CRÍTICA A ENTERO
        labels = data_imput[target_column].astype(float).astype(int)

        # SELECCIÓN DE MÉTODO
        metodo = st.sidebar.radio("Selecciona Modelo de IA", ["Random Forest (RF)", "Logistic Regression (LR)"])
        model_key = "RF" if "Random" in metodo else "LR"
        model_path = f"src/models/model_lluvia_{model_key}.pkl"

        # BOTÓN ENTRENAR
        if st.sidebar.button(f"🏋🧠 Entrenar {model_key}"):
            train_classifier_model(features, labels, model_type=model_key, model_path=model_path)
            st.sidebar.success(f"Modelo {model_key} listo para predecir lluvia.")

        # BOTÓN EVALUAR
        if st.sidebar.button("🔍📊 Comparar Rendimiento"):
            model, y_pred = train_classifier_model(features, labels, model_type=model_key, model_path=model_path)
            y_true = labels.iloc[int(len(labels) * 0.8):]
            metrics = evaluate_classification(y_true, y_pred)
            st.write(f"### Métricas de {metodo}")
            st.json(metrics)

        # SECCIÓN DE PREDICCIÓN
        st.markdown("---")
        st.header("🔮 Predicción Semanal")
        if st.button("¿Va a llover la próxima semana?"):
            if os.path.exists(model_path):
                model = joblib.load(model_path)
                # Tomamos los últimos datos del dataset para predecir el futuro inmediato
                predicciones = model.predict(features.tail(7))
                
                # Mostrar resultados visuales
                cols = st.columns(7)

                # Mostramos los resultados de forma elegante
                for i, p in enumerate(predicciones):
                    dia_num = i + 1
                    if p == 1:
                        st.info(f"📅 **Día {dia_num}:** Sí, **va a llover** 🌧️ (Se necesita el paraguas)")
                    else:
                        st.success(f"📅 **Día {dia_num}:** No llovera, **estará seco** ☀️ (No hace falta el paraguas)")
            else:
                st.error("Error: Primero debes entrenar el modelo seleccionado en el menú lateral.")

if __name__ == "__main__":
    main()