@echo off
setlocal

cd /d "%~dp0"

where py >nul 2>&1
if %errorlevel%==0 (
    set "PYTHON_CMD=py -3"
) else (
    where python >nul 2>&1
    if %errorlevel%==0 (
        set "PYTHON_CMD=python"
    ) else (
        echo Python was not found. Install Python 3.8+ and retry.
        pause
        exit /b 1
    )
)

if not exist ".venv\Scripts\python.exe" (
    echo Creating virtual environment...
    %PYTHON_CMD% -m venv .venv
    if errorlevel 1 goto :failed
)

set "VENV_PY=.venv\Scripts\python.exe"

echo Installing dependencies...
"%VENV_PY%" -m pip install -r requirements.txt
if errorlevel 1 goto :failed

set "APP_URL=http://127.0.0.1:5000"

where powershell >nul 2>&1
if %errorlevel%==0 (
    start "" powershell -NoProfile -ExecutionPolicy Bypass -Command "Start-Sleep -Seconds 3; Start-Process '%APP_URL%'"
) else (
    start "" "%APP_URL%"
)

echo Starting Doc2JSON backend...
"%VENV_PY%" app.py
if errorlevel 1 goto :failed

exit /b 0

:failed
echo Failed to start Doc2JSON.
pause
exit /b 1
