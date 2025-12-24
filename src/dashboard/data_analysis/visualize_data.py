import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

# from src.utils.p_value import correlation_pvalue_matrix
from src.utils.p_value_2 import correlation_heatmap

def visualization_Data (data):
    with st.expander("📊 Visualización de Datos"):
        # correlation_pvalue_matrix(data)
        correlation_heatmap(data)

        # 3. Distribución de variables numéricas
        with st.expander("📈 Distribuciones"):
            st.subheader("📈 Distribuciones")
            columna = st.selectbox("Selecciona una columna numérica:", data.select_dtypes(include=["float", "int"]).columns)
            fig, ax = plt.subplots()
            sns.histplot(data[columna], bins=30, kde=True, ax=ax)
            st.pyplot(fig)

        # 4. Series temporales
        with st.expander("⏳ Evolución temporal"):
            st.subheader("⏳ Evolución temporal")
            col_time = st.selectbox("Selecciona variable para graficar en el tiempo:", ["tmed", "prec", "hrmedia"])
            if "fecha" in data.columns:
                fig, ax = plt.subplots()
                data.groupby("fecha")[col_time].mean().plot(ax=ax)
                ax.set_title(f"Evolución de {col_time} en el tiempo")
                st.pyplot(fig)