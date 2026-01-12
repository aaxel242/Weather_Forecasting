import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import os

def visualization_Data(data):
    """
    Renderiza dashboards PowerBI e gráficos interactivos: histogramas, comparación temporal, scatter y series.
    Parámetros: data (DataFrame). Retorna: None (renderiza visualizaciones en Streamlit).
    """

    # Definimos la ruta de la carpeta
    image_path = "src/images/"

    # Lista de imágenes con sus títulos y descripciones personalizadas
    # Asegúrate de que los nombres de archivo coincidan exactamente con los que tienes en la carpeta
    imagenes_info = [
        {
            "archivo": "pbi_temp.png", 
            "titulo": "Dashboard General",
            "info": "Resumen global de temperaturas y precipitaciones anuales."
        },
        {
            "archivo": "pbi_lluvia.png", 
            "titulo": "Análisis de precipitación",
            "info": "Relación humedad y precipitación."
        }
    ]

    # Creamos columnas (en este caso 3 columnas para que estén una al lado de la otra)
    cols = st.columns(len(imagenes_info))

    for i, img_data in enumerate(imagenes_info):
        with cols[i]:
            # Título encima de la imagen
            st.markdown(f"### {img_data['titulo']}")
            
            # Ruta completa del archivo
            full_path = os.path.join(image_path, img_data['archivo'])
            
            # Verificamos si la imagen existe antes de mostrarla
            if os.path.exists(full_path):
                st.image(full_path, width="stretch")
                # Información debajo de la imagen
                st.caption(img_data['info'])
            else:
                st.error(f"No se encontró: {img_data['archivo']}")

    st.divider()

    interactive_distribution(data)

def interactive_distribution(data):
    """
    Genera 4 gráficos interactivos: histograma, comparación temporal, scatter y serie temporal.
    Parámetros: data (DataFrame). Retorna: None (renderiza gráficos con Streamlit y Matplotlib).
    """
    
    # Preparamos las columnas numéricas
    cols_numericas = data.select_dtypes(include=[np.number]).columns.tolist()

    # Layout de 2x2
    row1_col1, row1_col2 = st.columns(2)
    row2_col1, row2_col2 = st.columns(2)

    # 1. HISTOGRAMA
    with row1_col1:
        st.subheader("1. Distribución")
        feat_hist = st.selectbox("Selecciona variable:", cols_numericas, key="hist")
        # El spinner muestra una barra de carga local
        with st.spinner('Actualizando Histograma...'):
            fig1, ax1 = plt.subplots()
            sns.histplot(data[feat_hist], kde=True, color="skyblue", ax=ax1)
            ax1.set_title(f"Distribución de {feat_hist}")
            st.pyplot(fig1)

    # 2. COMPARACIÓN ENTRE COLUMNAS
    with row1_col2:
        st.subheader("2. Comparación entre columnas (Step Plot)")

        feat_a = st.selectbox("Variable A:", cols_numericas, key="step_a")
        feat_b = st.selectbox("Variable B:", cols_numericas, key="step_b")

        with st.spinner('Generando gráfico escalonado...'):

            # Preparamos dataframe para graficar
            if "date" in data.columns:
                df_plot = data.set_index("date")[[feat_a, feat_b]]
            elif "fecha" in data.columns:
                df_plot = data.set_index("fecha")[[feat_a, feat_b]]
            else:
                df_plot = data[[feat_a, feat_b]]

            fig, ax = plt.subplots(figsize=(8, 4))

            ax.step(df_plot.index, df_plot[feat_a], label=feat_a, linewidth=1.2, where="mid")
            ax.step(df_plot.index, df_plot[feat_b], label=feat_b, linewidth=1.2, where="mid")

            ax.set_title(f"Comparación temporal (Step Plot): {feat_a} vs {feat_b}")
            ax.set_xlabel("Tiempo")
            ax.set_ylabel("Valores")
            ax.legend()

            plt.xticks(rotation=45)
            st.pyplot(fig)

    # 3. SCATTER PLOT
    with row2_col1:
        st.subheader("3. Relación X vs Y")
        feat_x = st.selectbox("Eje X:", cols_numericas, index=0, key="scat_x")
        feat_y = st.selectbox("Eje Y:", cols_numericas, index=1, key="scat_y")
        with st.spinner('Dibujando Dispersión...'):
            fig3, ax3 = plt.subplots()
            sns.scatterplot(data=data, x=feat_x, y=feat_y, alpha=0.5, ax=ax3)
            ax3.set_title(f"{feat_x} frente a {feat_y}")
            st.pyplot(fig3)

    # 4. EVOLUCIÓN TEMPORAL (Solución al error de 'date')
    with row2_col2:
        st.subheader("4. Evolución Temporal")
        feat_time = st.selectbox("Variable temporal:", cols_numericas, key="time")
        
        with st.spinner('Cargando Serie Temporal...'):
            # SOLUCIÓN: Si 'date' es el índice, lo recuperamos con .index
            fig4, ax4 = plt.subplots()
            
            # Usamos data.index porque limpiar_datos hizo set_index('date')
            ax4.plot(data.index, data[feat_time], color="orange", linewidth=0.5)
            
            ax4.set_title(f"Serie de tiempo: {feat_time}")
            plt.xticks(rotation=45)
            st.pyplot(fig4)