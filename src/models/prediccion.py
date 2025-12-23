import joblib
import streamlit as st
import os
from datetime import datetime, timedelta
import pandas as pd

def predict_with_model(model_path_lr, model_path_rf, features, rango):
    st.markdown("---")
    st.header(f"🌦️ Predicción Meteorológica - {rango} Días")
    
    if os.path.exists(model_path_rf) and os.path.exists(model_path_lr):
        model_rf = joblib.load(model_path_rf)
        model_lr = joblib.load(model_path_lr)
        
        predicciones_rf = model_rf.predict(features.tail(rango))
        predicciones_lr = model_lr.predict(features.tail(rango))
        
        # Crear fechas para los próximos 7 días
        fecha_inicio = datetime.now()
        fechas = [fecha_inicio + timedelta(days=i) for i in range(rango)]
        
        # TAB 1: Random Forest
        tab1, tab2 = st.tabs(["🌳 Random Forest", "📊 Regresión Logística"])
        
        with tab1:
            st.subheader("Pronóstico del Modelo Random Forest")
            cols = st.columns(7)
            
            for idx, (col, prediccion, fecha) in enumerate(zip(cols, predicciones_rf, fechas)):
                with col:
                    dia_nombre = fecha.strftime("%a").upper()
                    dia_num = fecha.strftime("%d")
                    
                    if prediccion == 1:
                        st.markdown(f"""
                        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                                    padding: 20px; border-radius: 15px; text-align: center; 
                                    color: white; box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>
                            <h2 style='margin: 0;'>🌧️</h2>
                            <p style='margin: 5px 0; font-weight: bold;'>{dia_nombre}</p>
                            <p style='margin: 5px 0; font-size: 18px;'>{dia_num}</p>
                            <p style='margin: 5px 0; font-size: 12px;'>Lluvia esperada</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div style='background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                                    padding: 20px; border-radius: 15px; text-align: center; 
                                    color: white; box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>
                            <h2 style='margin: 0;'>☀️</h2>
                            <p style='margin: 5px 0; font-weight: bold;'>{dia_nombre}</p>
                            <p style='margin: 5px 0; font-size: 18px;'>{dia_num}</p>
                            <p style='margin: 5px 0; font-size: 12px;'>Sin lluvia</p>
                        </div>
                        """, unsafe_allow_html=True)
        
        # TAB 2: Regresión Logística
        with tab2:
            st.subheader("Pronóstico del Modelo Regresión Logística")
            cols = st.columns(7)
            
            for idx, (col, prediccion, fecha) in enumerate(zip(cols, predicciones_lr, fechas)):
                with col:
                    dia_nombre = fecha.strftime("%a").upper()
                    dia_num = fecha.strftime("%d")
                    
                    if prediccion == 1:
                        st.markdown(f"""
                        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                                    padding: 20px; border-radius: 15px; text-align: center; 
                                    color: white; box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>
                            <h2 style='margin: 0;'>🌧️</h2>
                            <p style='margin: 5px 0; font-weight: bold;'>{dia_nombre}</p>
                            <p style='margin: 5px 0; font-size: 18px;'>{dia_num}</p>
                            <p style='margin: 5px 0; font-size: 12px;'>Lluvia esperada</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div style='background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                                    padding: 20px; border-radius: 15px; text-align: center; 
                                    color: white; box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>
                            <h2 style='margin: 0;'>☀️</h2>
                            <p style='margin: 5px 0; font-weight: bold;'>{dia_nombre}</p>
                            <p style='margin: 5px 0; font-size: 18px;'>{dia_num}</p>
                            <p style='margin: 5px 0; font-size: 12px;'>Sin lluvia</p>
                        </div>
                        """, unsafe_allow_html=True)
        
        # RESUMEN ESTADÍSTICO
        st.markdown("---")
        st.subheader("📈 Estadísticas del Pronóstico")
        
        col1, col2, col3, col4 = st.columns(4)
        
        dias_lluvia_rf = sum(predicciones_rf)
        dias_lluvia_lr = sum(predicciones_lr)
        
        with col1:
            st.metric("☔ Días lluvia (RF)", f"{dias_lluvia_rf}/{rango}")
        with col2:
            st.metric("☀️ Días seco (RF)", f"{rango-dias_lluvia_rf}/{rango}")
        with col3:
            st.metric("☔ Días lluvia (LR)", f"{dias_lluvia_lr}/{rango}")
        with col4:
            st.metric("☀️ Días seco (LR)", f"{rango-dias_lluvia_lr}/{rango}")
        
    else:
        st.error("❌ Error: Primero debes entrenar el modelo seleccionado en el menú lateral.")