import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

def render_eda_section():
    st.header("🔍 Análisis Exploratorio de Datos Históricos")
    
    # --- CORRECCIÓN DE RUTA PARA PROFUNDIDAD EXTRA ---
    # Estamos en: src/dashboard/data_analysis/eda.py
    # 1. dirname -> src/dashboard/data_analysis
    # 2. dirname -> src/dashboard
    # 3. dirname -> src
    base_src = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # Construimos la ruta hacia los datos
    csv_path = os.path.join(base_src, 'data', 'processed', 'data_weather_oficial.csv')

    # Cargar datos
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        st.error(f"❌ Error: No se encuentra el archivo en:\n{csv_path}")
        return

    # 1. Muestra de Datos
    st.subheader("Vista de Datos")
    st.dataframe(df.head())
    
    col1, col2 = st.columns(2)
    
    # 2. Estadísticas Básicas
    with col1:
        st.markdown("### Estadísticas Descriptivas")
        cols_existentes = [c for c in ['tmax', 'tmin', 'prec', 'velmedia'] if c in df.columns]
        st.write(df[cols_existentes].describe())

    # 3. Distribución
    with col2:
        st.markdown("### Distribución de Temperaturas Máximas")
        if 'tmax' in df.columns:
            fig, ax = plt.subplots()
            sns.histplot(df['tmax'], kde=True, color="orange", ax=ax)
            ax.set_title("Histograma T_MAX")
            st.pyplot(fig)

    # 4. Correlaciones
    st.markdown("### Mapa de Correlaciones (Heatmap)")
    fig_corr, ax_corr = plt.subplots(figsize=(10, 6))
    
    posibles_cols = ['tmax', 'tmin', 'prec', 'dewpoint_2m_c_mean', 'cloudcover__mean', 'surface_pressure_hpa_mean']
    cols_corr = [c for c in posibles_cols if c in df.columns]
    
    if len(cols_corr) > 1:
        sns.heatmap(df[cols_corr].corr(), annot=True, cmap='coolwarm', ax=ax_corr)
        st.pyplot(fig_corr)
    else:
        st.info("No hay suficientes columnas numéricas para el heatmap.")