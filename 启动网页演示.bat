@echo off
cd /d "%~dp0"
echo Starting web demo ...
echo.
set "PY=D:\python\python3.14.5\python.exe"
if not exist "%PY%" set "PY=python"
"%PY%" start_demo.py
echo.
pause
