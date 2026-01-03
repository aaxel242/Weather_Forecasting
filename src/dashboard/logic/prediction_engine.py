import streamlit as st
import joblib
import pandas as pd
import os

@st.cache_resource(show_spinner="Cargando Cerebro IA...")
def cargar_modelos(base_path):
    """Carga los modelos .pkl y retorna None si fallan"""
    try:
        models_dir = os.path.join(base_path, 'models')
        m_tmax = joblib.load(os.path.join(models_dir, 'modelo_tmax.pkl'))
        m_tmin = joblib.load(os.path.join(models_dir, 'modelo_tmin.pkl'))
        m_lluvia = joblib.load(os.path.join(models_dir, 'modelo_lluvia.pkl'))
        return m_tmax, m_tmin, m_lluvia
    except Exception as e:
        st.error(f"Error cargando modelos: {e}")
        return None, None, None

def preparar_datos_prediccion(base_path):
    """Lee el CSV y prepara las últimas 7 filas para predecir"""
    try:
        csv_path = os.path.join(base_path, 'data', 'processed', 'data_weather_oficial.csv')
        df = pd.read_csv(csv_path)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        
        # Ingeniería de características básica necesaria para los modelos
        if 'tmax' in df.columns: df['tmax_yesterday'] = df['tmax'].shift(1)
        if 'tmin' in df.columns: df['tmin_yesterday'] = df['tmin'].shift(1)
        df['mes'] = df['date'].dt.month
        df['dia_anio'] = df['date'].dt.dayofyear
        
        # Tomamos los últimos 7 días (simulando forecast)
        return df.tail(7).copy().reset_index(drop=True)
    except Exception as e:
        st.error(f"Error leyendo datos: {e}")
        return None

def ejecutar_predicciones(df, m_tmax, m_tmin, m_lluvia):
    """Ejecuta los modelos y devuelve arrays con resultados"""
    if df is None or not m_tmax:
        return None, None, None
        
    # Obtener columnas requeridas por el modelo (seguridad)
    f_tmax = getattr(m_tmax, "feature_names_in_", df.columns)
    f_tmin = getattr(m_tmin, "feature_names_in_", df.columns)
    
    # Fallback manual para features de lluvia si no están definidos en el pickle
    f_rain_default = ['cloudcover__mean', 'cloudcover__max', 'surface_pressure_hpa_min', 
                    'surface_pressure_hpa_mean', 'hrmedia', 'hrmax', 'mes', 'dia_anio']
    f_rain = getattr(m_lluvia, "feature_names_in_", f_rain_default)
    
    # Filtrar solo columnas existentes para evitar crash
    f_rain = [c for c in f_rain if c in df.columns]

    try:
        p_tmax = m_tmax.predict(df[f_tmax])
        p_tmin = m_tmin.predict(df[f_tmin])
        p_rain = m_lluvia.predict(df[f_rain])
        return p_tmax, p_tmin, p_rain
    except Exception as e:
        st.error(f"Error en predicción: {e}")
        return None, None, None