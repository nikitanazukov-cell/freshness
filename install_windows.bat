@echo off
setlocal
cd /d "%~dp0"
echo Creating an isolated Python 3.13 environment...
where py >nul 2>&1
if errorlevel 1 goto no_python
py -3.13 -m venv .venv
if errorlevel 1 goto no_python
".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 goto failed
".venv\Scripts\python.exe" -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cpu
if errorlevel 1 goto failed
".venv\Scripts\python.exe" -m pip install -r requirements-core.txt
if errorlevel 1 goto failed
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto failed
".venv\Scripts\python.exe" main.py doctor
if errorlevel 1 goto failed
echo.
echo Installation completed. Read README.md before training.
pause
exit /b 0
:no_python
echo Install Python 3.13 x64 with Python Launcher and Tcl/Tk.
echo See README.md for manual setup with another supported Python version.
pause
exit /b 1
:failed
echo Installation failed. Read the error above and docs\TROUBLESHOOTING_RU.md.
pause
exit /b 1
