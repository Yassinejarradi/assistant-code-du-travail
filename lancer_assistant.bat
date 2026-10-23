@echo off
rem One-click launcher: starts Ollama if needed, then the web app (opens the browser).
rem Close this window (or press Ctrl+C) to stop the app.
title Assistant Code du travail - ENSAM
cd /d "%~dp0"

rem Model caches live on D: (C: is almost full); offline mode skips Hugging Face network checks.
set HF_HOME=D:\ai-cache\huggingface
set OLLAMA_MODELS=D:\ai-cache\ollama
rem Ollama 0.40's default CUDA 13 runtime no longer supports the GTX 1050 Ti (compute 6.1) and
rem silently falls back to the CPU; CUDA 12 still supports it (prompt reading ~12x faster).
set OLLAMA_LLM_LIBRARY=cuda_v12
set HF_HUB_OFFLINE=1
set HF_HUB_DISABLE_SYMLINKS_WARNING=1

curl -s http://localhost:11434/api/version >nul 2>&1
if errorlevel 1 (
    echo Demarrage d'Ollama...
    start "Ollama" /min "%LOCALAPPDATA%\Programs\Ollama\ollama.exe" serve
    timeout /t 5 /nobreak >nul
)

echo Lancement de l'assistant sur http://localhost:8501 ...
echo (le premier chargement des modeles prend environ 1 minute)
".venv\Scripts\python.exe" -m streamlit run app.py
pause
