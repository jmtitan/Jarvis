@echo off
:: Jarvis Voice Assistant Windowed Launcher
:: Purpose: Activates conda environment and starts the application with visible console window

title Jarvis Voice Assistant
color 0B

:: Get parent directory
cd /d "%~dp0.."

:: Check conda availability
where conda >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    color 0C
    echo Error: Conda not found. Please install Anaconda or Miniconda.
    pause
    exit /b 1
)

:: Initialize conda for cmd.exe if needed
call conda --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    :: Find conda location and initialize it
    for /f "delims=" %%i in ('where conda') do set "CONDA_EXE=%%i"
    set "CONDA_ROOT=%CONDA_EXE:\Scripts\conda.exe=%"
    call "%CONDA_ROOT%\Scripts\activate.bat"
)

:: Activate the conda environment
echo Activating conda environment: jarvis
call conda activate jarvis
if %ERRORLEVEL% NEQ 0 (
    color 0C
    echo Error: Failed to activate jarvis environment.
    echo Make sure it exists by running: conda create -n jarvis python=3.8
    pause
    exit /b 1
)

:: Start Jarvis with visible console
echo Starting Jarvis...
echo.
echo Note: This window will remain open to show debug output.
echo Close this window to terminate Jarvis.
echo.
echo ========== Jarvis Debug Output ==========
echo.

python src/main.py

echo.
echo Jarvis has terminated.
pause 