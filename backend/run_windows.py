"""Inicia o backend do IAbel no Windows (use o Python do venv: venv\\Scripts\\python.exe)."""
import os
import sys

# O console do Windows não usa UTF-8 por padrão e o backend imprime emojis
os.environ.setdefault("PYTHONUTF8", "1")
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BACKEND_DIR)
sys.path.insert(0, BACKEND_DIR)

import uvicorn

uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=False)
