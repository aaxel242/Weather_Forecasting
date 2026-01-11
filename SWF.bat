@echo off
title Lanzador App Clima
setlocal

:: 1. Entramos en la carpeta del proyecto
cd /d "%~dp0"

:: 2. Forzamos a Streamlit a ejecutarse desde la carpeta raiz
:: Usamos comillas dobles para que el espacio en "Proyecto 3" no rompa el comando

call uv run python -m streamlit run src/main.py

pause