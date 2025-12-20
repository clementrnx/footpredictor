@echo off
SETLOCAL EnableDelayedExpansion
title iTrOz Predictor - Full Installer

echo ==========================================
echo    iTrOz Predictor : AUTO-INSTALLER
echo ==========================================

python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [!] Python n'est pas detecte. 
    echo [>] Telechargement de Python 3.12...
    curl -L https://www.python.org/ftp/python/3.12.2/python-3.12.2-amd64.exe -o python_installer.exe
    
    echo [>] Installation silencieuse...
    start /wait python_installer.exe /quiet InstallAllUsers=1 PrependPath=1
    del python_installer.exe
    echo [OK] Python installe.
) else (
    echo [OK] Python est deja present.
)

echo [>] Installation des modules (Streamlit, Numpy, Scipy, Requests)...
python -m pip install --upgrade pip
python -m pip install streamlit requests numpy scipy

echo ==========================================
echo ✅ INSTALLATION TERMINEE
echo ==========================================
pause
