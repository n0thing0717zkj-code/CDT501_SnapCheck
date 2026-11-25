@echo off
cd /d "%~dp0"
set PY="D:\python\python3.14.5\python.exe"
echo [1/5] check.py
%PY% check.py
if errorlevel 1 goto fail
echo [2/5] make_dataset.py --pages 60 --per-page 5
%PY% make_dataset.py --pages 60 --per-page 5
if errorlevel 1 goto fail
echo [3/5] train_model.py
%PY% train_model.py
if errorlevel 1 goto fail
echo [4/5] run_experiments.py
%PY% run_experiments.py
if errorlevel 1 goto fail
echo [5/5] make_figures.py
%PY% make_figures.py
echo.
echo  All done. See out\summary.csv and out\figures
pause
exit /b 0
:fail
echo.
echo  Stopped: previous step failed.
pause
exit /b 1
