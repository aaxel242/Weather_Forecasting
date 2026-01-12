import streamlit as st

def basic_stats (data):
    # Muestra estadísticas básicas del dataset: primeras filas, descripción y métricas de calidad.
    # Parámetro: data (DataFrame). Muestra resumen de filas, columnas, nulos y duplicados.
    st.subheader("Dataset")
    st.dataframe(data.head(100), width="stretch")

    st.divider()

    st.subheader("Datos básicos")
    st.write(data.describe(include="all"))

    st.divider()

    st.subheader("Revisión del dataset")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Filas", len(data))
    col2.metric("Columnas", len(data.columns))
    col3.metric("Valores nulos", int(data.isna().sum().sum()))
    col4.metric("Filas duplicadas", int(data.duplicated().sum()))