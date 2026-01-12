# 🌤️ Weather Forecasting AI

Un **sistema inteligente de predicción meteorológica** que combina aprendizaje automático con análisis de datos históricos para pronosticar temperatura máxima, temperatura mínima y precipitación para los próximos 7 días. Incluye un dashboard interactivo con recomendaciones personalizadas según las condiciones climáticas predichas.

---

## Tabla de Contenidos

- [Características Principales](#características-principales)
- [Requisitos Previos](#requisitos-previos)
- [Instalación](#instalación)
- [Uso](#uso)
- [Arquitectura del Proyecto](#arquitectura-del-proyecto)
- [Modelos de Aprendizaje Automático](#modelos-de-aprendizaje-automático)
- [Dataset](#dataset)
- [Resultados y Métricas](#resultados-y-métricas)
- [Estructura de Carpetas](#estructura-de-carpetas)
- [Tecnologías Utilizadas](#tecnologías-utilizadas)
- [Contribuciones](#contribuciones)
- [Licencia](#licencia)

---

## Características Principales

### Predicciones Meteorológicas
- **Temperatura Máxima**: Modelo RandomForest entrenado con histórico completo de datos meteorológicos
- **Temperatura Mínima**: Predicción con lags temporales y características estacionales
- **Precipitación**: Clasificación binaria (lluvia/sin lluvia) con umbral ajustado (0.35 de probabilidad)

### Dashboard Interactivo
- Interfaz visual intuitiva con tarjetas animadas (efecto flip)
- Muestra predicciones para los próximos 7 días
- Iconografía dinámica según condiciones climáticas
- Recomendaciones personalizadas (qué ropa llevar, si llevar paraguas, etc.)

### Análisis Exploratorio de Datos (EDA)
- Estadísticas descriptivas del dataset
- Matriz de correlaciones con p-values
- Visualizaciones interactivas de distribuciones
- Evaluación comparativa de modelos

### Predicciones Recursivas
- Usa valores predichos como entrada para próximas predicciones (forecasting de 7 días)
- Implementa lags temporales para capturar patrones estacionales
- Manejo inteligente de presión atmosférica como predictor de tormentas

---

## Requisitos Previos

- **Python**: ≥ 3.10
- **pip**: Sistema de gestión de paquetes de Python
- **Git**: Para clonar el repositorio (opcional)

---

## Instalación

### 1. Clonar el repositorio
```bash
git clone https://github.com/tu-usuario/Weather_Forecasting.git
cd Weather_Forecasting
```

### 2. Crear un entorno virtual
```bash
python -m venv .venv
```

### 3. Activar el entorno virtual
**En Windows (PowerShell):**
```powershell
.\.venv\Scripts\Activate.ps1
```

**En Windows (CMD):**
```cmd
.venv\Scripts\activate.bat
```

**En macOS/Linux:**
```bash
source .venv/bin/activate
```

### 4. Instalar dependencias

**Opción A: Usar pip (método recomendado)**
```bash
pip install -r requirements.txt
```

**Opción B: Usar pip con pyproject.toml**
```bash
pip install -e .
```

**Opción C: Usar uv (más rápido y eficiente)**
```bash
uv sync
```

### 5. Estructura de datos requerida
Asegúrate de que existen los archivos:
```
src/data/processed/
  └── data_weather_final.csv  (Dataset con features pre-procesadas)
```

---

## Uso

### Archivo SWF.bat

```
Enviar archvio como acceso directo al escritorio o ejecutar directamente
Este hace:
1.- call uv sync
2.- call uv run python -m streamlit run src/main.py
```


### Dashboard Streamlit (Recomendado)

```bash
streamlit run src/dashboard/app.py
```

El dashboard se abrirá en tu navegador (generalmente en `http://localhost:8501`)

## Arquitectura del Proyecto

### Flujo de Datos
```
Datos Históricos (CSV)
    ↓
[Limpieza → Imputación → Features Engineered]
    ↓
[Entrenamiento de Modelos]
    ↓ (offline)
Dashboard Streamlit
    ↓
[Cargar Modelos → Preparar Features → Predicciones Recursivas]
    ↓
[Renderizar Tarjetas + Recomendaciones]
    ↓
Usuario
```

### Componentes Principales

#### 1. **Ingesta y Preparación de Datos** (`src/data/`)
- `unir_json.py`: Unifica múltiples archivos JSON/TXT en CSV
- `add_lags.py`: Ingeniería de features (lags, estacionalidad, targets a 7 días)
- `compilar.py`: Compila datos en directorios organizados

#### 2. **Modelos de Predicción** (`src/models/`)
- `train_model_temp_max.py`: RandomForest para temperatura máxima
- `train_model_temp_min.py`: RandomForest para temperatura mínima
- `train_model_precipitation.py`: RandomForest+SMOTE para lluvia (clasificación binaria)
- `evaluation.py`: Métricas de evaluación
- `comparate.py`: Comparación de algoritmos (RF vs LR vs SVM)

#### 3. **Motor de Predicción** (`src/scripts/`)
- `prediction_engine.py`: 
  - `cargar_modelos()`: Carga modelos y features desde disco
  - `preparar_datos_prediccion()`: Prepara datos históricos
  - `ejecutar_predicciones()`: Loop recursivo de 7 días con lags dinámicos
- `eda.py`: Sección de análisis exploratorio en dashboard

#### 4. **Utilidades** (`src/utils/`)
- `cargar_datos.py`: Carga CSV inicial
- `limpieza.py`: Limpia datos (outliers, valores inválidos)
- `imputar_datos.py`: Imputa NaNs mediante interpolación
- `data_analysis.py`: Estadísticas descriptivas
- `visualize_data.py`: Gráficos exploratorios
- `p_value.py`: Matriz de correlaciones con p-values
- `show_evaluation.py`: Tabla de métricas de modelos
- `recommendations.py`: Lógica de recomendaciones personalizadas

#### 5. **Dashboard UI** (`src/dashboard/`)
- `app.py`: Función principal que orquesta todo
- `ui/cards.py`: Genera tarjetas HTML animadas con flip effect
- `ui/styles.py`: Estilos CSS personalizados (tema oscuro)

---

## Modelos de Aprendizaje Automático

### Temperatura Máxima y Mínima (Regresión)
- **Algoritmo**: Random Forest Regressor
- **Features clave**:
  - Lags: `tmax_lag1`, `tmin_lag1`, `prec_lag1` (día anterior)
  - Estacionalidad: mes, día del año, estación
  - Meteorología: punto de rocío, nubosidad, humedad relativa
  - Presión: cambio de presión respecto al día anterior

- **Métricas**:
  - MAE (Error Medio Absoluto) ~1.5-2.0 °C
  - R² Score ~0.85-0.90
  - RMSE (Raíz del Error Cuadrático Medio)

### Precipitación (Clasificación Binaria)
- **Algoritmo**: RandomForest + SMOTE (manejo de desbalance) + GrindSearch (mejora de prediccion)
- **Target**: `bin_prep` (1 = lluvia, 0 = sin lluvia)
- **Features especiales**:
  - `pressure_delta`: Cambio de presión (mejor predictor de tormentas)
  - `pressure_yesterday`: Presion atmosférica del día anterior
  - `rain_yesterday_bin`: Lluvia el día anterior
  - Delta de presión negativo = mayor probabilidad de lluvia

- **Métricas**:
  - Accuracy: ~80-85%
  - Recall: ~70-75% (detecta la mayoría de lluvias)
  - Precision: ~60-65% (pocos falsos positivos)
  - F1-Score: ~0.65-0.70

### Modelos Comparados
Se evaluaron 3 algoritmos:
1. **Random Forest** (Mejor rendimiento general)
2. **Regresión Logística** (Baseline)
3. **SVM** (Útil para comparación)

---

## Dataset

### Fuentes de Datos
- **Museo Marítimo de Barcelona**: Datos 2009-2025 (múltiples archivos por semestre)
- **Puerto Olímpico**: Datos 2023-2025
- **OneWeather**: Dataset complementario 2024

### Variables Principales
| Variable | Descripción | Tipo |
|----------|-------------|------|
| `date` | Fecha de observación | DateTime |
| `tmax` | Temperatura máxima (°C) | Float |
| `tmin` | Temperatura mínima (°C) | Float |
| `prec` | Precipitación (mm) | Float |
| `surface_pressure_hpa_mean` | Presión atmosférica media (hPa) | Float |
| `cloudcover__mean` | Cobertura nubosa media (%) | Float |
| `hrmedia` | Humedad relativa media (%) | Float |
| `dewpoint_2m_c_mean` | Punto de rocío medio (°C) | Float |

### Pre-procesamiento
1. **Limpieza**: Eliminación de outliers estadísticos
2. **Imputación**: Interpolación lineal y media para NaNs
3. **Feature Engineering**:
   - Lags: 1, 2, 3, 7 días
   - Estacionalidad: mes, día del año, estación
   - Targets: tmax, tmin, prec a 7 días en el futuro
4. **División temporal**: Train/Test sin mezclar (respeta cronología)

---

## 📈 Resultados y Métricas

### Temperatura Máxima
```
MAE:     1.87 °C
RMSE:    2.34 °C
R²:      0.876
```

### Temperatura Mínima
```
MAE:     1.62 °C
RMSE:    2.08 °C
R²:      0.891
```

### Precipitación (Clasificación)
```
Accuracy:  81.2%
Precision: 60.2%
Recall:    72.2%
F1-Score:  0.6567
```
---

## Estructura de Carpetas

```
Weather_Forecasting/
├── src/
│   ├── data/
│   │   ├── raw/                      # Datos crudos originales
│   │   │   ├── Barcelona - Museo Maritimo/
│   │   │   ├── Barcelona - Puerto Olimpico/
│   │   │   └── datos de oneweather/
│   │   ├── processed/                # Datos procesados y listos
│   │   │   ├── data_weather_final.csv
│   │   │   ├── data_weather_oficial.csv
│   │   │   └── data_binario.csv
│   │   ├── add_lags.py              # Ingeniería de features
│   │   ├── unir_json.py             # Unificación de datos
│   │   └── compilar.py              # Compilación de CSV
│   │
│   ├── models/                       # Modelos entrenados y scripts
│   │   ├── modelo_tmax.joblib       # Modelo Random Forest temp máx
│   │   ├── modelo_tmin.joblib       # Modelo Random Forest temp mín
│   │   ├── modelo_lluvia.joblib     # Modelo precipitación
│   │   ├── features_*.joblib        # Listas de features por modelo
│   │   ├── train_model_temp_max.py
│   │   ├── train_model_temp_min.py
│   │   ├── train_model_precipitation.py
│   │   ├── evaluation.py            # Métricas
│   │   ├── comparate.py             # Comparación de algoritmos
│   │   └── prediccion.py            # Funciones de predicción (legacy)
│   │
│   ├── scripts/
│   │   ├── prediction_engine.py     # ⭐ Motor de predicciones recursivas
│   │   └── eda.py                   # Análisis exploratorio
│   │
│   ├── utils/
│   │   ├── cargar_datos.py
│   │   ├── limpieza.py
│   │   ├── imputar_datos.py
│   │   ├── data_analysis.py
│   │   ├── visualize_data.py
│   │   ├── p_value.py
│   │   ├── show_evaluation.py
│   │   └── recommendations.py
│   │
│   ├── dashboard/
│   │   ├── app.py                   # ⭐ Función principal Streamlit
│   │   ├── ui/
│   │   │   ├── cards.py             # Tarjetas con flip animation
│   │   │   └── styles.py            # Estilos CSS
│   │   └── images/                  # Iconos y imágenes
│   │
│   └── main.py                      # Punto de entrada alternativo
│
├── .venv/                           # Entorno virtual
├── pyproject.toml                   # Dependencias del proyecto
├── requirements.txt                 # Lista de paquetes
├── README.md                        # Este archivo
└── SWF.bat                          # Script de inicialización (Windows)
```

---

## Tecnologías Utilizadas

### Backend y ML
| Tecnología | Versión | Propósito |
|-----------|---------|----------|
| **pandas** | ≥2.0.0 | Manipulación de DataFrames |
| **numpy** | ≥1.24.0 | Operaciones numéricas |
| **scikit-learn** | ≥1.3.0 | Modelos ML (RandomForest, GridSearchCV) |
| **imbalanced-learn** | ≥0.14.1 | SMOTE para desbalance de clases |
| **joblib** | ≥1.4.0 | Serialización de modelos |

### Frontend
| Tecnología | Versión | Propósito |
|-----------|---------|----------|
| **Streamlit** | ≥1.52.1 | Dashboard interactivo |
| **matplotlib** | ≥3.10.8 | Gráficos estáticos |
| **seaborn** | ≥0.13.2 | Visualización estadística |
| **plotly** | ≥6.5.0 | Gráficos interactivos |

### Desarrollo
| Tecnología | Propósito |
|-----------|----------|
| **python-dotenv** | Gestión de variables de entorno |
| **pydantic** | Validación de datos |
| **pytest** | Testing (opcional) |
| **ruff** | Linting y formateo |

---

## Variables de Entorno

Si necesitas usar variables de entorno, crea un archivo `.env`:

```env
# Rutas de datos
DATA_PATH=src/data/processed/
MODELS_PATH=src/models/

# Configuración de Streamlit
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_HEADLESS=true
```

---

## Decisiones de Diseño

### 1. Predicciones Recursivas
Se implementó un loop de 7 días donde cada predicción usa la anterior como lag:
- **Ventaja**: Captura patrones a mediano plazo
- **Desafío**: Acumulación de errores
- **Solución**: Uso de presión atmosférica como predictor físico principal

### 2. SMOTE para Desbalance
El dataset tiene ~20% de días lluviosos vs ~80% secos:
- **Problema**: Modelo tendría sesgos
- **Solución**: SMOTE sobremuestrea la clase minoritaria
- **Pipeline**: SMOTE + RandomForest dentro de GridSearchCV

### 3. Sin mezclar Train/Test
Se usó `shuffle=False` en el split:
- **Razón**: Datos son serie temporal
- **Ventaja**: Simula predicción real (futuro desconocido)
- **Métrica**: Temporal Train/Test Split respeta cronología

---

## Troubleshooting

### Error: "No se encuentra modelo_tmax.joblib"
- **Causa**: No has entrenado los modelos
- **Solución**: Ejecuta `python src/models/train_model_temp_max.py`

### Error: "ModuleNotFoundError"
- **Causa**: No activaste el entorno virtual
- **Solución**: Ejecuta `.\.venv\Scripts\Activate.ps1` (Windows)

### Error: "CSV no encontrado"
- **Causa**: Falta `src/data/processed/data_weather_final.csv`
- **Solución**: Ejecuta `python src/data/add_lags.py`

### Dashboard lento
- **Causa**: Los modelos tardán en cargar/predecir
- **Solución**: Se usa `@st.cache_resource` para cachear modelos
- **Alternativa**: Aumenta RAM o reduce el tamaño del dataset

---

## Documentación Adicional

### Archivos de Referencia
- [Tabla de Datos](src/data/processed/TablaDatos.md): Descripción detallada de variables
- [README de datos raw](src/data/raw/README.md): Fuentes y enlaces

### Próximas Mejoras
- [ Que nuestro modelo de precipitación nos de un resultado acorde a la realidad] 
- [ Actualizar la BDD constantemente] 
- [ Mejorar algunos aspectos del frontend] 
- [ Hacerlo público a personas] 
- [ ] 
- [ ] 

---

## Contacto y Soporte

Para reportar bugs o sugerir mejoras:
- Abre un Issue en GitHub
- Contacta al equipo de desarrollo
- Conectarse con un profesorado del Stucom

---

## Agradecimientos

- Inspirado en proyectos de forecasting meteorológico de código abierto
- Profesorado del master IABD Stucom

---

**Última actualización**: Enero 2026  
**Versión**: 0.1.0  
**Estado**: En desarrollo activo