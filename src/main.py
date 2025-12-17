import streamlit as st
import os
from PIL import Image

from utils.cargar_datos import cargar_datos
from utils.compilar import compilar_datos
from utils.limpieza import limpiar_datos
from utils.imputar_datos import imputar_datos

from scripts.eda import eda_interactivo

from models.prediccion import load_model, predict_with_model
from models.train_model import train_classifier_model, train_regression_model 
from models.evaluation import evaluate_classification, evaluate_regression

def main():
    # Imagen de inicio
    image_path = os.path.join("src", "images", "imagen_entrada.png")
    if os.path.exists(image_path):
        image = Image.open(image_path) 
        st.image(image)

    #######################
    # Sidebar: definir datos de entrada 
    st.sidebar.header("📅 Opciones de entrada")
    fecha = st.sidebar.date_input("Selecciona la fecha")
    rango = st.sidebar.slider("Rango de días a predecir", 1, 7, 3)

    st.sidebar.markdown("---")

    # st.sidebar.header("🗃️📊 Datos")
    # if st.sidebar.button("📦 Compilar datos"):
        # compilar_datos()

    # Cargar datos --> Subprograma en utils
    data = cargar_datos() 
  
    # Limpiar datos automáticamente
    if data is not None:
        data_clean = limpiar_datos(data)        
        data_imput = imputar_datos(data_clean)  
               
        eda_interactivo(data_imput)
        
        # Botón opcional para guardar CSV limpio
        if st.sidebar.button("💾 Guardar datos limpios en CSV"):
            ruta_csv_limpio = os.path.join("src", "datos", "csv", "datos_clima_limpios.csv")
            data_imput.to_csv(ruta_csv_limpio, index=False)
            st.sidebar.success(f"✅ Datos limpios guardados en: {ruta_csv_limpio}")
            
    st.sidebar.markdown("---")

    # if st.sidebar.button("🏋🧠 Entrenar modelo"):
    #     df_col = [load_data['tmed'], load_data['tmin'], load_data['tmax'], load_data['prec'], load_data['dir'], load_data['velmedia'],
    #               load_data['racha'], load_data['hrmedia'], load_data['hrmax'], load_data['hrmin']]
    #     entrenar_modelo(load_data, df_col)

    # st.sidebar.markdown("---")
        
    # if st.sidebar.button("🔍📊 Evaluar rendimiento"):
    #     evaluar_modelo()

    # #######################

    # # Botón de predicción
    # if data_imput is not None and st.button("Generar predicción"):
    #     st.info(f"Prediciendo clima desde {fecha} durante {rango} días...")
    #     resultado = predecir_clima(data_imput, fecha, rango)  # Subprograma en models 
    #     st.write(resultado)

    target_column = "tmed" # Objetivo continuo: requiere REGRESIÓN

    if data_imput is not None and target_column in data_imput.columns:
        # Separar Features y Labels antes de entrenar
        features = data_imput.drop(columns=[target_column], errors='ignore')
        labels = data_imput[target_column]

        if st.sidebar.button("🏋🧠 Entrenar modelo"):
            # 1. CAMBIO AQUÍ: Se usa train_regression_model para predecir 'tmed'
            model, _ = train_regression_model( # <--- ¡CORREGIDO!
                features, labels, model_path="src/models/trained_regression_tmed_model.pkl"
            )
            st.sidebar.success(f"Modelo de Regresión entrenado y guardado para '{target_column}'.")

        if st.sidebar.button("🔍📊 Evaluar rendimiento"):
            st.sidebar.info(f"Evaluando modelo para la variable '{target_column}' (Regresión)...")
            
            # Para la evaluación, ejecutamos el entrenamiento para OBTENER las predicciones de prueba (y_pred)
            # 2. CAMBIO AQUÍ: Se usa train_regression_model para obtener las predicciones
            model, y_pred_test = train_regression_model( # <--- ¡CORREGIDO!
                features, labels, model_path="src/models/trained_regression_tmed_model.pkl"
            )
            
            # Obtenemos los valores reales de prueba (y_true)
            split_factor = 0.8
            split = int(len(labels) * split_factor)
            y_true_test = labels.iloc[split:]

            # 3. Se usa evaluate_regression con los datos correctos
            regression_metrics = evaluate_regression(y_true_test, y_pred_test)

            st.sidebar.subheader(f"Métricas de Regresión ({target_column})")
            st.sidebar.json(regression_metrics)
            
            st.sidebar.warning("Se omitió la evaluación de Clasificación, ya que 'tmed' es una variable continua.")


        if st.button("Generar predicción"):
            # Se carga el modelo de regresión
            model = load_model("src/models/trained_regression_tmed_model.pkl")
            # Para la predicción, se debe pasar el dataframe sin la columna 'tmed'
            resultado = predict_with_model(model, features) 
            st.write("Predicción generada (usando todas las filas como features):")
            st.write(resultado)

if __name__ == "__main__":
    main()
