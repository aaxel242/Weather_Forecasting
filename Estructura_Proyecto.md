# Estructura completa para proyecto

* Escogemos lo necesario --> Se ha hecho contemplando todas las situaciones, pueden no ser necesarias en su desarrollo.

WEATHER_FORECASTING/
│
├── .streamlit/              # Configuración de Streamlit
│   ├── config.toml         # Tema, configuración del servidor
│   └── secrets.toml        # API keys (NO versionar en git)
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── external/
│
├── notebooks/
│   ├── exploratory/
│   └── reports/
│
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── load_data.py
│   │   └── preprocess.py
│   │
│   ├── features/
│   │   ├── __init__.py
│   │   └── build_features.py
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── train.py
│   │   └── predict.py
│   │
│   ├── visualization/
│   │   ├── __init__.py
│   │   └── visualize.py
│   │
│   └── utils/
│       ├── __init__.py
│       └── helpers.py
│
├── app/                     # 🆕 Todo lo relacionado con Streamlit
│   ├── __init__.py
│   ├── main.py             # Aplicación principal de Streamlit
│   ├── pages/              # Páginas múltiples de Streamlit
│   │   ├── 01_📊_Dashboard.py
│   │   ├── 02_🔮_Predictions.py
│   │   ├── 03_📈_Historical_Data.py
│   │   └── 04_ℹ️_About.py
│   │
│   ├── components/         # Componentes reutilizables de UI
│   │   ├── __init__.py
│   │   ├── charts.py
│   │   ├── metrics.py
│   │   └── sidebar.py
│   │
│   └── styles/            # CSS personalizado
│       └── custom.css
│
├── assets/                 # 🆕 Recursos estáticos
│   ├── images/            # Imágenes, logos, iconos
│   │   ├── logo.png
│   │   ├── banner.jpg
│   │   └── weather_icons/
│   │
│   ├── fonts/             # Fuentes personalizadas (opcional)
│   └── animations/        # GIFs, Lottie files, etc.
│
├── models/
│   └── trained/           # Modelos entrenados
│
├── reports/
│   └── figures/
│
├── tests/
│   ├── __init__.py
│   ├── test_data.py
│   └── test_models.py
│
├── config/
│   ├── config.yaml        # Configuración general
│   └── model_config.yaml  # Configuración de modelos
│
├── scripts/
│   ├── train_model.py
│   ├── make_predictions.py
│   └── download_data.py   # 🆕 Script para obtener datos
│
├── docs/                   # 🆕 Documentación del proyecto
│   ├── api.md
│   ├── setup.md
│   └── user_guide.md
│
├── .github/                # 🆕 CI/CD (opcional)
│   └── workflows/
│       └── tests.yml
│
├── .gitignore
├── .python-version
├── pyproject.toml          # 🆕 Configuración moderna (uv usa esto)
├── uv.lock                 # 🆕 Lock file de uv
├── README.md
├── LICENSE
└── Makefile               # 🆕 Comandos útiles (opcional)