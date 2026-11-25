@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo Dang kiem tra thu vien...
python -m pip install -r requirements.txt --quiet --disable-pip-version-check
python app.py
pause
