@echo off
echo ====================================================
echo STARTING AI PROJECT ENVIRONMENT SETUP
echo ====================================================

REM Check if venv folder already exists
if not exist "venv\" (
    echo [1/4] venv folder not found. Creating new virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment. Please ensure Python is installed and in PATH.
        pause
        exit /b 1
    )
    echo Virtual environment venv created successfully.
) else (
    echo [1/4] venv folder already exists. Skipping creation step.
)

REM Activate virtual environment
echo [2/4] Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip to latest version
echo [3/4] Upgrading pip to latest version...
python -m pip install --upgrade pip

REM Install dependencies from requirements.txt
echo [4/4] Installing dependencies from requirements.txt (including PyTorch CPU)...
pip install -r requirements.txt

echo ====================================================
echo SETUP COMPLETED SUCCESSFULLY!
echo ====================================================
echo To start working or running scripts, activate the virtual environment using:
echo     venv\Scripts\activate
echo.
pause
