import streamlit as st
import prophet as prophet

from utils.generate_correlation import correlation_matrix, generate_plots
from utils.calculate_p_value import correlation_pvalue_matrix

def correlation(data):

    st.markdown("----------------------")
    st.title("1 Correlation Data Plot")
    st.subheader("Correaltion Matrix")

    # mostramos los datos originales en un dataframe
    st.dataframe(data.head())

    # verficamos que todas las columnas sean de tipo numérico y las converitmos en una lista
    numeric_cols = data.select_dtypes(include=["number"]).columns.tolist()

    # validamos del número mínimo de columnas numéricas
    if len(numeric_cols) < 2:
        st.error("Dataset must contain at least 2 numeric columns.")
        return

    # le pasamos los datos, las columnas y le decimos que nos devuelva la correlacion de los datos
    corr = correlation_matrix(data,columnas=numeric_cols)

    p_values = []
    p_values = correlation_pvalue_matrix(data)

    with st.expander("Valores p"):
        st.dataframe(p_values, use_container_width=True)

    # prediction_model(data)
    with st.expander("Scatter Plots"):
        generate_plots(data,corr, numeric_cols)

    return data