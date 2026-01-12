import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os
from datetime import timedelta


"""
    Carga los modelos y mostrar las predicciones para los próximos 7 días.
"""

@st.cache_resource
def cargar_modelos(base_path):
    """
    Carga modelos entrenados (.joblib) y listas de features desde carpeta 'models'.
    Parámetros: base_path (str ruta raíz del proyecto).
    Retorna: tuplas (modelo, features) para tmax, tmin, lluvia.
    """
    try:
        # CORRECCIÓN DE RUTA: Apuntamos directamente a 'models' dentro del base_path
        models_dir = os.path.join(base_path, 'models')
        
        # Cargar Modelos (Objetos entrenados)
        m_tmax = joblib.load(os.path.join(models_dir, 'modelo_tmax.joblib'))
        m_tmin = joblib.load(os.path.join(models_dir, 'modelo_tmin.joblib'))
        m_lluvia = joblib.load(os.path.join(models_dir, 'modelo_lluvia.joblib')) 
        
        # Cargar Listas de Features (Columnas exactas usadas en el entrenamiento)
        f_tmax = joblib.load(os.path.join(models_dir, 'features_tmax.joblib'))
        f_tmin = joblib.load(os.path.join(models_dir, 'features_tmin.joblib'))
        f_lluvia = joblib.load(os.path.join(models_dir, 'features_lluvia.joblib'))
        
        return (m_tmax, f_tmax), (m_tmin, f_tmin), (m_lluvia, f_lluvia)
    except Exception as e:
        st.error(f"Error crítico cargando modelos: {e}")
        return None, None, None

def preparar_datos_prediccion(base_path):
    """
    Carga CSV histórico completo para obtener el punto de partida de predicciones.
    Parámetros: base_path (str ruta raíz).
    Retorna: DataFrame ordenado cronológicamente.
    """
    try:
        # CORRECCIÓN DE RUTA: Ajustamos para llegar a data/processed/
        csv_path = os.path.join(base_path, 'data', 'processed', 'data_weather_final.csv')
        
        if not os.path.exists(csv_path):
            st.error(f"No se encuentra el archivo CSV en: {csv_path}")
            return None

        df = pd.read_csv(csv_path)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        
        # Devolvemos el DF completo, la lógica de selección se hace en ejecutar_predicciones
        return df 
    except Exception as e:
        st.error(f"Error leyendo datos históricos: {e}")
        return None

def ejecutar_predicciones(df_historico, pack_tmax, pack_tmin, pack_lluvia):
    """
    Genera predicciones recursivas para 7 días usando lógica de lags.
    Parámetros: df_historico (DataFrame base), packs de modelos (modelo, features).
    Retorna: listas (tmax, tmin, lluvia) para próximos 7 días.
    """
    # Desempaquetar tuplas (Modelo, Lista de Features)
    m_tmax, feats_tmax = pack_tmax
    m_tmin, feats_tmin = pack_tmin
    m_lluvia, feats_lluvia = pack_lluvia
    
    # 1. ESTADO INICIAL (Semilla)
    # Cogemos la última fila de datos reales conocidos
    last_day = df_historico.iloc[-1]
    
    # Variables dinámicas (se actualizarán en cada vuelta del bucle)
    current_tmax = last_day['tmax']
    current_tmin = last_day['tmin']
    current_prec = last_day['prec']
    current_date = pd.to_datetime(last_day['date'])
    
    # Datos estáticos o de persistencia (copiamos el resto de columnas del último día)
    # En un sistema real, aquí consultaríamos una API externa para nubes/viento futuros.
    # Aquí asumimos que condiciones como 'dewpoint' o 'cloudcover' varían poco o usamos persistencia.
    static_data = last_day.to_dict() 
    
    predictions = {'fechas': [], 'tmax': [], 'tmin': [], 'rain': []}

    # 2. BUCLE DE PREDICCIÓN (7 DÍAS)
    for i in range(1, 8):
        next_date = current_date + timedelta(days=i)
        
        # Preparamos el diccionario de inputs para este día futuro
        input_dict = static_data.copy()
        
        # --- LOGICA DE PRESIÓN ---
        # Como no tenemos un modelo para predecir la presión futura, asumimos estabilidad.
        # pressure_delta = 0.0 (Sin cambio brusco). 
        # Si tuvieras datos, aquí pondrías (presion_hoy - presion_ayer).
        pressure_delta_simulado = 0.0 

        # Actualizamos las variables temporales y los lags (recursividad)
        input_dict.update({
            'dia_anio': next_date.dayofyear,
            'mes': next_date.month,
            
            # LAGS: Lo que ocurrió "ayer" (que es la iteración anterior del bucle)
            'tmax_yesterday': current_tmax,
            'tmin_yesterday': current_tmin,
            'prec_yesterday': current_prec,
            'rain_yesterday_bin': 1 if current_prec > 0.1 else 0,
            
            # NUEVA VARIABLE (Si entrenaste el modelo de lluvia con ella)
            'pressure_delta': pressure_delta_simulado
        })
        
        # Convertimos a DataFrame de 1 fila
        input_row = pd.DataFrame([input_dict])

        # --- PREDICCIONES ---
        
        # 1. Temperatura Máxima
        # Filtramos input_row para pasarle SOLO las columnas que este modelo aprendió
        p_tmax = m_tmax.predict(input_row[feats_tmax])[0]
        
        # 2. Temperatura Mínima
        p_tmin = m_tmin.predict(input_row[feats_tmin])[0]
        
        # 3. Lluvia (Con Umbral Ajustado)
        # Usamos predict_proba para ver la "confianza" del modelo
        try:
            probs = m_lluvia.predict_proba(input_row[feats_lluvia])[0]
            prob_rain = probs[1] # Probabilidad de la clase 1 (Lluvia)
            
            # APLICAMOS UMBRAL MANUAL (0.35 = 35%)
            # Si el modelo tiene más de 35% de certeza, decimos que llueve.
            # Esto mejora el Recall (detecta más lluvias reales).
            p_rain_bin = 1 if prob_rain >= 0.35 else 0
        except:
            # Fallback por si el modelo no soporta predict_proba
            p_rain_bin = m_lluvia.predict(input_row[feats_lluvia])[0]
        
        # Guardamos resultados
        predictions['fechas'].append(next_date)
        predictions['tmax'].append(p_tmax)
        predictions['tmin'].append(p_tmin)
        predictions['rain'].append(p_rain_bin)
        
        # --- ACTUALIZACIÓN DE ESTADO ---
        # Los valores predichos hoy se convierten en el "ayer" de la siguiente vuelta
        current_tmax = p_tmax
        current_tmin = p_tmin
        
        # Si predecimos lluvia, asumimos una cantidad pequeña (ej: 2.0mm) para el lag de mañana.
        # Si no, 0.0. Esto ayuda a la coherencia de la serie temporal.
        current_prec = 2.0 if p_rain_bin == 1 else 0.0 
        
        current_date = next_date

    return predictions['tmax'], predictions['tmin'], predictions['rain']