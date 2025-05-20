@echo off
:: Jarvis Voice Assistant Launcher
:: Purpose: Activates conda environment and starts the application

:: Get parent directory
cd /d "%~dp0.."

:: Check conda availability
where conda >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Error: Conda not found. Please install Anaconda or Miniconda.
    pause
    exit /b 1
)

:: Activate the conda environment
call conda activate jarvis
if %ERRORLEVEL% NEQ 0 (
    echo Error: Failed to activate jarvis environment.
    echo Make sure it exists by running: conda create -n jarvis python=3.8
    pause
    exit /b 1
)

:: Start Jarvis
python src/main.py

exit /b 