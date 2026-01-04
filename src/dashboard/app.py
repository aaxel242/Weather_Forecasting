import streamlit as st
import os

# Importamos módulos
from dashboard.ui.styles import apply_custom_styles
from dashboard.ui.cards import generar_grid_html
from dashboard.logic.prediction_engine import cargar_modelos, preparar_datos_prediccion, ejecutar_predicciones
from dashboard.data_analysis.eda import render_eda_section

def main_frontend():
    """
    Función principal que orquesta la UI y la Lógica.
    """
    
    # 1. Aplicar estilos globales (Fondo, scrollbars)
    apply_custom_styles()
    
    # 2. Header

    # imagen inicio
    col_izq, col_centro, col_der = st.columns([1, 1, 1])
    with col_centro:
        st.image("src/images/imagen_entrada.png")

    st.markdown("<h2 style='text-align: center; color: #f8fafc;'>🌦️ Predicción Meteorológica IA</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #94a3b8; margin-bottom: 1rem;'>Desliza el ratón sobre los días para ver recomendaciones</p>", unsafe_allow_html=True)

    # 3. Lógica del Backend
    base_src = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    with st.spinner("Consultando modelos meteorológicos..."):
        # A. Cargar Modelos
        m_tmax, m_tmin, m_lluvia = cargar_modelos(base_src)
        
        # B. Preparar Datos
        df_futuro = preparar_datos_prediccion(base_src)
        
        # C. Predecir y Renderizar
        if df_futuro is not None and m_tmax is not None:
            # Ejecutamos modelos
            p_tmax, p_tmin, p_rain = ejecutar_predicciones(df_futuro, m_tmax, m_tmin, m_lluvia)
            
            # --- CAMBIO IMPORTANTE AQUÍ ---
            # Llamamos a la función. Ya no recogemos un string HTML,
            # la función renderiza el componente directamente.
            generar_grid_html(df_futuro, p_tmax, p_tmin, p_rain, base_src)
            
        else:
            st.error("⚠️ Error: No se pudieron cargar los modelos o los datos.")
            st.info(f"Ruta base detectada: {base_src}")


    # Seleccion de expansiones
    
    st.markdown("---")

    col_1, col_2= st.columns([1, 1])

    # 4. Sección EDA
    with col_1 :
        
        with st.expander("📊 Ver Análisis de Datos Históricos"):
            render_eda_section()
    
    with col_2 :
        with st.expander("Ubicación de las estaciones Meteorológica"):
            st.markdown("Museo Maritimo")
            st.image("src/images/ubicacion_estación__museo_maritimo.png")
            st.markdown("---")
            st.markdown("Puerto Olimpico")
            st.image("src/images/ubicacion_estación__puerto_olimpico.png")
