@echo off
rem Prepara o IAbel no Windows: Python 3.12 (via uv), dependencias do backend e do frontend
cd /d "%~dp0"
set UV_LINK_MODE=copy

where uv >nul 2>nul
if errorlevel 1 (
    echo Instalando uv...
    python -m pip install --user uv || goto :erro
    for /f "delims=" %%i in ('python -c "import uv; print(uv.find_uv_bin())"') do set "UV=%%i"
) else (
    set "UV=uv"
)

echo === Backend: Python 3.12 + venv ===
"%UV%" python install 3.12
if not exist "backend\venv\Scripts\python.exe" (
    "%UV%" venv backend\venv --python 3.12 --seed || goto :erro
)

echo === Backend: dependencias ===
"%UV%" pip install --python backend\venv\Scripts\python.exe torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu || goto :erro
"%UV%" pip install --python backend\venv\Scripts\python.exe -r backend\requirements-windows.txt || goto :erro

if not exist "backend\.env" (
    copy backend\.env.example backend\.env >nul
    echo ATENCAO: backend\.env criado a partir do exemplo. Preencha a GOOGLE_API_KEY.
)

echo === Frontend: dependencias ===
pushd frontend
call npm ci --no-audit --no-fund || (popd & goto :erro)
popd

echo.
echo Instalacao concluida. Para iniciar: iniciar_windows.bat
pause
exit /b 0

:erro
echo.
echo ERRO durante a instalacao. Veja as mensagens acima.
pause
exit /b 1
