@echo off
setlocal

cd /d "%~dp0"

call :resolve_python
if errorlevel 1 (
    echo Python 3.8+ was not found. Installing Python with winget...
    where winget >nul 2>&1
    if errorlevel 1 (
        echo winget is not available on this system.
        echo Install Python manually from https://www.python.org/downloads/windows/ and run this script again.
        pause
        exit /b 1
    )

    winget install --id Python.Python.3.12 -e --source winget --scope user --accept-package-agreements --accept-source-agreements
    if errorlevel 1 (
        echo Automatic Python installation failed.
        pause
        exit /b 1
    )

    call :resolve_python
    if errorlevel 1 (
        call :resolve_python_from_localappdata
        if errorlevel 1 (
            echo Python was installed, but it is not available in this shell yet.
            echo Close this window and run doc2json-windows.bat again.
            pause
            exit /b 1
        )
    )
)

if not exist ".venv\Scripts\python.exe" (
    echo Creating virtual environment...
    call "%PYTHON_EXE%" %PYTHON_FLAGS% -m venv .venv
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

:resolve_python
where py >nul 2>&1
if %errorlevel%==0 (
    set "PYTHON_EXE=py"
    set "PYTHON_FLAGS=-3"
    call :validate_python
    if not errorlevel 1 exit /b 0
)

where python >nul 2>&1
if %errorlevel%==0 (
    set "PYTHON_EXE=python"
    set "PYTHON_FLAGS="
    call :validate_python
    if not errorlevel 1 exit /b 0
)

set "PYTHON_EXE="
set "PYTHON_FLAGS="
exit /b 1

:resolve_python_from_localappdata
for /f "delims=" %%D in ('dir /b /ad "%LocalAppData%\Programs\Python\Python*" 2^>nul ^| sort /r') do (
    if exist "%LocalAppData%\Programs\Python\%%D\python.exe" (
        set "PYTHON_EXE=%LocalAppData%\Programs\Python\%%D\python.exe"
        set "PYTHON_FLAGS="
        call :validate_python
        if not errorlevel 1 exit /b 0
    )
)

set "PYTHON_EXE="
set "PYTHON_FLAGS="
exit /b 1

:validate_python
call "%PYTHON_EXE%" %PYTHON_FLAGS% -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 8) else 1)" >nul 2>&1
if errorlevel 1 exit /b 1
exit /b 0
