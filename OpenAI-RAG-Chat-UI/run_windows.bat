@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  python -m venv .venv
  if errorlevel 1 (echo Could not create virtual environment. & pause & exit /b 1)
)
".venv\Scripts\python.exe" -m pip install --upgrade pip
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 (echo Dependency installation failed. & pause & exit /b 1)
if not exist ".env" (
  copy ".env.example" ".env" >nul
  notepad ".env"
  echo Add your OpenAI API key, save .env, then run this file again.
  pause
  exit /b 0
)
".venv\Scripts\python.exe" -m streamlit run app.py
pause
