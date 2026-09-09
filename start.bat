@echo off
setlocal
cd /d "%~dp0"
chcp 65001 >nul
set "PYTHONUTF8=1"
title FreeTokenAPI
echo Starting FreeTokenAPI...

if exist ".venv\Scripts\python.exe" goto check_dependencies

echo Creating a local Python virtual environment...
py -3 -c "import sys; sys.exit(sys.version_info < (3, 10))" >nul 2>nul
if errorlevel 1 goto try_python
py -3 -m venv ".venv"
if errorlevel 1 goto failed
goto check_dependencies

:try_python
python -c "import sys; sys.exit(sys.version_info < (3, 10))" >nul 2>nul
if errorlevel 1 goto python_missing
python -m venv ".venv"
if errorlevel 1 goto failed

:check_dependencies
".venv\Scripts\python.exe" -c "import fastapi, uvicorn, httpx, dotenv; from pydantic import field_validator; from PIL import Image" >nul 2>nul
if not errorlevel 1 goto run
echo Installing Python dependencies...
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto failed

:run
".venv\Scripts\python.exe" app.py
set "EXIT_CODE=%ERRORLEVEL%"
if not "%EXIT_CODE%"=="0" pause
exit /b %EXIT_CODE%

:python_missing
echo Python 3.10+ was not found. Install Python, add it to PATH, and try again.
pause
exit /b 1

:failed
echo Setup failed. Check the error above and the instructions in README.md.
pause
exit /b 1
