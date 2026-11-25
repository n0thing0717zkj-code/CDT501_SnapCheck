@echo off
cd /d "%~dp0"
set PY="D:\python\python3.14.5\python.exe"
echo ============================================
echo  DEMO 1/3 : clean phone photo (should ACCEPT)
echo ============================================
%PY% demo.py data\phone\p01a.jpg
echo.
pause
echo ============================================
echo  DEMO 2/3 : tilted phone photo (should REJECT)
echo ============================================
%PY% demo.py data\phone\p01b.jpg
echo.
pause
echo ============================================
echo  DEMO 3/3 : synthetic tilt (main cause = tilt)
echo ============================================
%PY% demo.py data\degraded\p050_tilt65_1.jpg
echo.
pause
