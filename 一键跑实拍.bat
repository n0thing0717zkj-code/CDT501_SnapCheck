@echo off
cd /d "%~dp0"
"D:\python\python3.14.5\python.exe" make_dataset.py --phone
echo.
echo  Done. See data\phone_labels.csv
pause
