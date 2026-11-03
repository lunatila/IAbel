@echo off
rem Inicia o backend (porta 8000) e o frontend (porta 3000) do IAbel no Windows
cd /d "%~dp0"

if not exist "backend\venv\Scripts\python.exe" (
    echo Ambiente do backend nao encontrado. Rode primeiro: instalar_windows.bat
    pause
    exit /b 1
)
if not exist "frontend\node_modules" (
    echo Dependencias do frontend nao encontradas. Rode primeiro: instalar_windows.bat
    pause
    exit /b 1
)

set PYTHONUTF8=1
start "IAbel Backend" cmd /k "cd /d "%~dp0backend" && venv\Scripts\python.exe run_windows.py"
start "IAbel Frontend" cmd /k "cd /d "%~dp0frontend" && npm run dev"

echo.
echo Backend:  http://localhost:8000  (docs: http://localhost:8000/docs)
echo Frontend: http://localhost:3000
echo O backend demora um pouco na primeira vez (carrega os modelos de embeddings).
timeout /t 15 >nul
start "" http://localhost:3000
