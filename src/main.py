import streamlit as st
import sys
import os

# --- 1. CONFIGURACIÓN DE PÁGINA ---
# IMPORTANTE: Esto debe ser siempre el primer comando de Streamlit
st.set_page_config(
    page_title="Weather AI",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- 2. CONFIGURACIÓN DEL PATH ---
# Añadimos la carpeta actual (src/) al path del sistema
# para que Python pueda encontrar el módulo 'dashboard'
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# --- 3. IMPORTAR Y EJECUTAR ---
try:
    # Ahora sí podemos importar dashboard porque el path está configurado
    from dashboard.app import main_frontend

    if __name__ == "__main__":
        main_frontend()
except ImportError as e:
    st.error("❌ Error Crítico de Importación")
    st.write(f"Detalle: {e}")
    st.warning("Asegúrate de que la carpeta 'dashboard' tiene un archivo __init__.py")
