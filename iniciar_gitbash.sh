#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [ -f .venv/Scripts/activate ]; then
  source .venv/Scripts/activate
elif [ -f .venv/bin/activate ]; then
  source .venv/bin/activate
else
  echo "No se encontró .venv. Lee README.md para preparar el entorno." >&2
  exit 1
fi
python -m streamlit run app.py
