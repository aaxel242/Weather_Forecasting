import streamlit as st
from src.utils.cargar_datos import cargar_datos
from src.utils.limpieza import limpiar_datos
from src.utils.imputar_datos import imputar_datos

# Importaciones corregidas
from src.dashboard.data_analysis.data_analysis import basic_stats
from src.dashboard.data_analysis.visualize_data import visualization_Data
from src.models.train_model import train_models
from models.train_model_precipitation import train_models
from src.models.evaluation import evaluate_classification

def eda_interactivo(data_imput):
    st.header("🔍 Exploratory Data Analysis (EDA)")
    
    with st.expander("Vista previa de la tabla de datos limpia e imputada"):
        st.dataframe(data_imput)
        basic_stats(data_imput)
        visualization_Data(data_imput)

    with st.expander("Evaluación de Modelos", expanded=True):
        # QUITAMOS el st.button anidado porque causa el error de reinicio
        # En su lugar, podemos usar un botón que ejecute la lógica directamente
        if st.button("🚀 Entrenar y Evaluar Modelos"):
            target_column_prep = "bin_prep"

            if data_imput is not None and target_column_prep in data_imput.columns:
                leaky = [target_column_prep, "date"]
                features = data_imput.drop(columns=[c for c in leaky if c in data_imput.columns], errors='ignore')
                labels = data_imput[target_column_prep].astype(float).astype(int)

                model_path = "src/models/"
                model_path_rf = f"{model_path}/model_lluvia_rf.pkl"
                model_path_lr = f"{model_path}/model_lluvia_lr.pkl"

                with st.spinner("Entrenando modelos..."):
                    model_rf, y_pred_rf, model_lr, y_pred_lr = train_models(features, labels, model_path_rf, model_path_lr)

                y_true = labels.iloc[int(len(labels) * 0.8):]
                
                # IMPORTANTE: Debes capturar el resultado y MOSTRARLO
                evaluate_classification(y_true, y_pred_rf, y_pred_lr)

# --- FLUJO PRINCIPAL ---
data = cargar_datos()
if data is not None:
    data_clean = limpiar_datos(data)
    data_imput = imputar_datos(data_clean)

    # RECOMENDACIÓN: Usa un checkbox en la barra lateral para mantener el estado
    mostrar_eda = st.sidebar.checkbox("Activar Análisis Exploratorio (EDA)")

    if mostrar_eda:
        eda_interactivo(data_imput)