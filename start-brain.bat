@echo off
setlocal
cd /d "%~dp0"
set "BRAIN_PYTHON=python"
if exist ".venv\Scripts\python.exe" set "BRAIN_PYTHON=%~dp0.venv\Scripts\python.exe"
"%BRAIN_PYTHON%" "%~dp0brain\brain2.py" %*
exit /b %errorlevel%
