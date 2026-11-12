@echo off
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" -m streamlit run app.py
) else (
  echo No se encontro el entorno .venv. Sigue el README para instalar las dependencias.
  pause
)
