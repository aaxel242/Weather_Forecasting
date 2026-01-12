# ============================================================================
# WEATHER FORECASTING - SISTEMA DE PREDICCIÓN METEOROLÓGICA CON IA
# ============================================================================
#
# ESTRUCTURA DEL PROYECTO:
#
# src/
#   ├── main.py                        # Punto de entrada principal de la aplicación
#   │
#   ├── dashboard/
#   │   ├── app.py                     # Orquestación principal del dashboard interactivo
#   │   └── ui/
#   │       ├── cards.py               # Componentes de tarjetas flip-card con predicciones
#   │       └── styles.py              # Estilos CSS personalizados de Streamlit
#   │
#   ├── models/                        # Modelos de Machine Learning
#   │   ├── train_model_temp_max.py    # Entrenamiento de RandomForest para temperatura máxima
#   │   ├── train_model_temp_min.py    # Entrenamiento de RandomForest para temperatura mínima
#   │   ├── train_model_precipitation.py # Entrenamiento de RandomForest+SMOTE para lluvia
#   │   ├── prediccion.py              # Funciones de predicción y lógica de agente meteorológico
#   │   ├── evaluation.py              # Evaluación de métricas de modelos
#   │   ├── comparate.py               # Comparación de múltiples algoritmos (RF, LR, SVM)
#   │   └── *.joblib                   # Modelos entrenados y features guardados
#   │
#   ├── scripts/
#   │   ├── prediction_engine.py       # Motor de predicción recursiva (7 días con lags)
#   │   └── eda.py                     # Análisis exploratorio de datos interactivo
#   │
#   ├── utils/                         # Funciones de utilidad general
#   │   ├── cargar_datos.py            # Carga de CSV históricos
#   │   ├── limpieza.py                # Limpieza y normalización de datos
#   │   ├── imputar_datos.py           # Imputación de valores faltantes
#   │   ├── data_analysis.py           # Estadísticas descriptivas básicas
#   │   ├── visualize_data.py          # Gráficos interactivos y dashboards
#   │   ├── show_evaluation.py         # Evaluación de modelos pre-entrenados
#   │   ├── recommendations.py         # Lógica de consejos y selección de iconos
#   │   └── p_value.py                 # Matriz de correlaciones y p-values
#   │
#   └── data/
#       ├── processed/                 # Datos limpios y procesados
#       │   └── data_weather_final.csv # Dataset definitivo para entrenamiento
#       └── raw/                       # Datos originales sin procesar
#
# ============================================================================
# FLUJO DE PREDICCIÓN:
# ============================================================================
#
# 1. USUARIO ABRE LA APP
#    └─> main.py llama a dashboard/app.py
#
# 2. DASHBOARD CARGA MODELOS
#    └─> prediction_engine.cargar_modelos() obtiene los 3 modelos (.joblib)
#
# 3. CARGA DATOS HISTÓRICOS
#    └─> prediction_engine.preparar_datos_prediccion() lee CSV base
#
# 4. GENERA PREDICCIONES (7 DÍAS)
#    └─> prediction_engine.ejecutar_predicciones()
#        • Usa último día real como punto de partida
#        • Predice tmax, tmin, lluvia para día +1
#        • Convierte predicciones en inputs para día +2 (lags recursivos)
#        • Repite para 7 días
#
# 5. RENDERIZA TARJETAS INTERACTIVAS
#    └─> ui/cards.py
#        • Crea 7 tarjetas flip-card con CSS 3D
#        • Anverso: icono del clima, temperaturas, indicador lluvia
#        • Reverso: recomendación personalizada (ropa, paraguas, etc)
#
# 6. USUARIO INTERACTÚA
#    └─> Datos analíticos en tab "EDA":
#        • Estadísticas
#        • Correlaciones
#        • Visualizaciones
#        • Métricas de modelos
#
# ============================================================================
# MODELOS ENTRENADOS:
# ============================================================================
#
# TEMPERATURA MÁXIMA (RandomForest - Regresión)
#   Features: tmax_yesterday, tmin_yesterday, día_año, mes, punto_rocío, nubosidad, etc.
#   Métrica: RMSE ~2-3°C, R² ~0.75-0.85
#
# TEMPERATURA MÍNIMA (RandomForest - Regresión)
#   Features: tmin_yesterday, tmax_yesterday, día_año, mes, nubosidad, punto_rocío, etc.
#   Métrica: RMSE ~2-3°C, R² ~0.70-0.80
#
# LLUVIA BINARIA (RandomForest + SMOTE - Clasificación)
#   Features: delta de presión, lluvia ayer, humedad, cobertura nubosa, estación, etc.
#   Métrica: F1-Score ~0.65-0.75, Recall ~0.70 (prioriza detectar lluvias)
#
# ============================================================================
# FLUJO DE LIMPIEZA DE DATOS:
# ============================================================================
#
# CSV RAW → limpiar_datos() → imputar_datos() → CSV FINAL
#
# limpiar_datos():
#   • Elimina duplicados
#   • Limpia comillas y espacios en blanco
#   • Normaliza nombres de columnas (lowercase, guiones bajos)
#   • Convierte a tipos numéricos
#   • Ordena cronológicamente
#
# imputar_datos():
#   • Variables estacionales (temp, humedad): media mensual
#   • Precipitación: asume 0 si falta
#   • Viento/Presión: interpolación lineal + media mensual
#   • Dirección viento: arrastre del valor anterior + media como fallback
#
# ============================================================================
# NOTAS IMPORTANTES:
# ============================================================================
#
# • LAGS RECURSIVOS: Las predicciones de hoy son inputs de mañana
#   Esto permite "caminar hacia el futuro" sin depender de APIs externas.
#
# • DELTA DE PRESIÓN: Mejor predictor físico de tormentas que cualquier otra variable.
#   Presión cayendo = probabilidad más alta de lluvia.
#
# • SMOTE: Técnica de sobremuestreo de la clase minoritaria (días lluvia) para
#   evitar que el modelo ignore las lluvias por ser menos frecuentes.
#
# • THRESHOLD MANUAL (0.35 o 35%): Se aplica en prediction_engine.py para
#   convertir probabilidades en predicción binaria. Ajustable según necesidades.
#
# ============================================================================
