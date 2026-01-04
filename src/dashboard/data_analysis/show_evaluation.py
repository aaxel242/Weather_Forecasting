import os
import streamlit as st
from src.models.evaluation import evaluate_classification
import joblib

def show_evaluation(data_imput):
    # --- SECCIÓN DE EVALUACIÓN (CARGA DIRECTA) ---
    st.divider()
    st.header("Evaluación de Modelos Pre-entrenados")
    
    target_col = "bin_prep"
    model_path_rf = "src/models/model_lluvia_RF.pkl"
    model_path_lr = "src/models/model_lluvia_LR.pkl"

    # Verificamos si los archivos existen antes de intentar cargarlos
    if os.path.exists(model_path_rf) and os.path.exists(model_path_lr):
        
        # Preparar los datos para la predicción (mismo proceso que en el entrenamiento)
        leaky = [target_col, "date", "fecha"]
        features = data_imput.drop(columns=[c for c in leaky if c in data_imput.columns], errors='ignore')
        labels = data_imput[target_col].astype(int)
        
        # Split para obtener el set de prueba (20%)
        split = int(len(features) * 0.8)
        X_test = features.iloc[split:]
        y_true = labels.iloc[split:]

        try:
            # CARGAR MODELOS
            model_rf = joblib.load(model_path_rf)
            model_lr = joblib.load(model_path_lr)

            # GENERAR PREDICCIONES
            y_pred_rf = model_rf.predict(X_test)
            y_pred_lr = model_lr.predict(X_test)

            # MOSTRAR EVALUACIÓN
            evaluate_classification(y_true, y_pred_rf, y_pred_lr)
            
        except Exception as e:
            st.error(f"Error al cargar o predecir con los modelos: {e}")
    else:
        st.warning("No se encontraron los archivos .pkl en 'src/models/'. Por favor, asegúrate de que los modelos estén entrenados y guardados.")