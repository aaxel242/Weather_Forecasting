import streamlit as st
import pandas as pd

def show_data(data: pd.DataFrame):
    st.header("Selección de fecha")

    # Detectar columna de fecha automáticamente
    col_fecha = "date" if "date" in data.columns else "fecha"

    # Convertir columna a tipo date
    fechas = pd.to_datetime(data[col_fecha]).dt.date

    # Selector de fecha
    fecha_seleccionada = st.date_input("Selecciona una fecha:", min(fechas))

    # Filtrar por fecha
    filtro = data[pd.to_datetime(data[col_fecha]).dt.date == fecha_seleccionada]

    if filtro.empty:
        st.warning(f"No hay datos disponibles para la fecha {fecha_seleccionada}")
        return

    st.subheader(f"Datos del {fecha_seleccionada}")

    # Convertir fila a dict
    fila = filtro.iloc[0].to_dict()

    # Mostrar valores en tarjetas (3 columnas)
    cols = st.columns(3)

    for i, (key, value) in enumerate(fila.items()):
        with cols[i % 3]:
            st.markdown(
                f"""
                <div style="
                    text-align:center; 
                    padding:15px; 
                    margin:15px; 
                    border-radius:10px; 
                    background-color:#f9f9f9; 
                    box-shadow:0 2px 6px rgba(0,0,0,0.1);
                ">
                    <h3 style="margin-bottom:8px; font-size:22px; color:#333;">{key}</h3>
                    <p style="margin:0; font-size:20px; color:#666;">{value}</p>
                </div>
                """,
                unsafe_allow_html=True
            )