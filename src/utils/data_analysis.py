import streamlit as st

from visualization.visualize_data import visualization_Data

def basic_stats (data):
    # 1. Resumen general
    with st.expander("📊 Resumen del dataset"):
        st.subheader("📊 Resumen del dataset")
        st.write(data.describe(include="all"))

        datos = {
            "n_rows": len(data),
            "n_columns": len(data.columns),
            "missing_values": int(data.isna().sum().sum()),
            "duplicated_rows": int(data.duplicated().sum()),
            }

        st.write("Estado básico del dataset:")
        for key, value in datos.items():
            st.info(f"{key}: {value}")