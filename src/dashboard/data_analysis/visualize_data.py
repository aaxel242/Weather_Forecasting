import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

def visualization_Data (data):
    st.title("📊 Visualización de Datos")

    # 3. Distribución de variables numéricas
    with st.expander("📈 Distribuciones"):
        st.subheader("📈 Distribuciones")
        columna = st.selectbox("Selecciona una columna numérica:", data.select_dtypes(include=["float", "int"]).columns)
        fig, ax = plt.subplots()
        sns.histplot(data[columna], bins=20, kde=True, ax=ax)
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