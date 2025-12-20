#!/bin/bash

echo "------------------------------------------"
echo "🏆 iTrOz Predictor : Installation Directe"
echo "------------------------------------------"

# 1. Vérification de Python3
if ! command -v python3 &> /dev/null
then
    echo "❌ Erreur : Python3 n'est pas détecté."
    exit
fi

# 2. Installation directe des modules
echo "🛠️ Installation des dépendances en cours..."
python3 -m pip install --upgrade pip
python3 -m pip install streamlit requests numpy scipy

# 3. Finalisation
echo "------------------------------------------"
echo "✅ Installation terminée."
echo "------------------------------------------"

# Lancement immédiat
read -p "Lancer iTrOz Predictor maintenant ? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]
then
    python3 -m streamlit run app.py
fi
