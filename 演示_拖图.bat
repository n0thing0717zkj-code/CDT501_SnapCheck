@echo off
cd /d "%~dp0"
if "%~1"=="" goto noarg
"D:\python\python3.14.5\python.exe" demo.py "%~1"
goto end
:noarg
echo.
echo   Drag a photo onto this file, then release.
echo.
:end
echo.
pause
