@echo off
title iTrOz Predictor Installer
echo ------------------------------------------
echo 🏆 iTrOz Predictor : Installation Windows
echo ------------------------------------------

:: Vérification de Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Erreur : Python n'est pas installe ou pas dans le PATH.
    pause
    exit
)

echo 🛠️ Installation des dependances...
python -m pip install --upgrade pip
python -m pip install streamlit requests numpy scipy

echo ------------------------------------------
echo ✅ Installation terminee.
echo ------------------------------------------
set /p launch="Lancer iTrOz Predictor maintenant ? (y/n) : "
if /i "%launch%"=="y" (
    python -m streamlit run app.py
)
pause
