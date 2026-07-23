@echo off
echo ========================================================
echo   TRAFFIC AI MARL PIPELINE LAUNCHER (2x2 Grid)
echo ========================================================
echo.
echo [1/2] Launching Streamlit Web Dashboard in Background...
start "Traffic AI Streamlit Dashboard" /min venv\Scripts\python.exe -m streamlit run dashboard\app.py --server.port 8501

echo [2/2] Starting MARL Training Loop (4 Independent Learners)...
venv\Scripts\python.exe agent\train_marl.py

echo.
echo Training complete or stopped!
pause
