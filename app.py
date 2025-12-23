import streamlit as st
import requests
import numpy as np
from scipy.stats import poisson
from datetime import datetime, timedelta
import pandas as pd

# --- CONFIGURATION CLEMENTRNXX PREDICTOR V13.0 ---
st.set_page_config(page_title="Clementrnxx Predictor V13.0", layout="wide")

DISCORD_WEBHOOK_URL = "https://discord.com/api/webhooks/1453026279275106355/gbYAwBRntm1FCoqoBTz5lj1SCe2ijyeHHYoe4CFYwpzOw2DO-ozcCsgkK_53HhB-kFGE"

st.markdown("""
    <style>
    .stApp { background-image: url("https://media.giphy.com/media/VZrfUvQjXaGEQy1RSn/giphy.gif"); background-size: cover; background-attachment: fixed; }
    .stApp > div:first-child { background-color: rgba(0, 0, 0, 0.96); }
    h1, h2, h3, p, span, label { color: #FFD700 !important; font-family: 'Monospace', sans-serif; }
    .section-box { border: 2px solid #FFD700; padding: 20px; border-radius: 15px; background: rgba(0,0,0,0.85); margin-bottom: 20px; }
    .audit-gold { border: 3px solid #FFD700; background: linear-gradient(145deg, #1a1a1a, #000); padding: 25px; border-radius: 15px; text-align: center; }
    .stake-val { font-size: 2.5rem; color: #00FF00 !important; font-weight: bold; text-shadow: 0 0 10px #00FF00; }
    div.stButton > button { background: #FFD700 !important; color: black !important; font-weight: 900; border-radius: 10px; height: 3.5em; width: 100%; }
    </style>
""", unsafe_allow_html=True)

# --- CONFIG GESTION DE RISQUE ( KELLY FRACTIONNAIRE ) ---
# Ici on définit la violence de la mise
STRATEGIES_MISE = {
    "ULTRA-SAFE (1%)": 0.01,   # On ne mise que 1% de la suggestion Kelly
    "SAFE (5%)": 0.05,
    "MID-SAFE (10%)": 0.10,
    "MID (20%)": 0.20,
    "AGRESSIF (50%)": 0.50,
    "ALL-IN (100%)": 1.00      # Kelly pur (très risqué, mise maximale)
}

# --- FONCTIONS TECHNIQUES ---
def calculate_probs(lh, la):
    matrix = np.zeros((10, 10))
    for x in range(10):
        for y in range(10): matrix[x, y] = poisson.pmf(x, lh) * poisson.pmf(y, la)
    matrix /= matrix.sum()
    return {
        "Home": np.sum(np.tril(matrix, -1)), "Draw": np.sum(np.diag(matrix)), "Away": np.sum(np.triu(matrix, 1)),
        "1N": np.sum(np.tril(matrix, -1)) + np.sum(np.diag(matrix)),
        "N2": np.sum(np.diag(matrix)) + np.sum(np.triu(matrix, 1)),
        "BTTS_Yes": np.sum(matrix[1:, 1:]), "BTTS_No": 1.0 - np.sum(matrix[1:, 1:])
    }

def get_team_stats(team_id, league_id, scope_overall):
    # Simulation simplifiée pour la structure
    return 1.6, 1.2 # scored, conceded

# --- SIDEBAR GESTIONNAIRE ---
with st.sidebar:
    st.header("💳 PORTEFEUILLE")
    bankroll = st.number_input("Capital Total (€)", 10.0, 1000000.0, 1000.0)
    mode_mise = st.selectbox("STRATÉGIE DE MISE", list(STRATEGIES_MISE.keys()), index=3)
    k_frac = STRATEGIES_MISE[mode_mise]
    
    st.warning(f"Attention : En mode {mode_mise}, l'algorithme calculera la mise optimale basée sur {k_frac*100}% de la formule de Kelly.")

tab1, tab2 = st.tabs(["🎯 ANALYSEUR 1VS1 & AUDIT", "📡 GÉNÉRATEUR DE TICKETS"])

with tab1:
    # --- 1. CONFIGURATION ---
    st.markdown("<div class='section-box'><h3>🛠 CONFIGURATION DU MATCH</h3>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    th = c1.text_input("Equipe Domicile")
    ta = c2.text_input("Equipe Extérieur")
    st.markdown("</div>", unsafe_allow_html=True)

    if th and ta:
        # Calcul probas (Exemple avec valeurs simulées pour l'interface)
        r = calculate_probs(1.7, 1.1) 
        
        # --- 2. ANALYSE ---
        st.markdown("<div class='section-box'><h3>📊 1. ANALYSE STATISTIQUE</h3>", unsafe_allow_html=True)
        m = st.columns(4)
        m[0].metric(th, f"{r['Home']:.1%}"); m[1].metric("NUL", f"{r['Draw']:.1%}"); m[2].metric(ta, f"{r['Away']:.1%}"); m[3].metric("BTTS", f"{r['BTTS_Yes']:.1%}")
        st.markdown("</div>", unsafe_allow_html=True)

        # --- 3. BET ---
        st.markdown("<div class='section-box'><h3>💰 2. ZONE DE BET</h3>", unsafe_allow_html=True)
        type_pari = st.selectbox("MARCHÉ CHOISI", ["Victoire Domicile", "Match Nul", "Victoire Extérieur", "1N", "N2", "BTTS OUI"])
        cote_book = st.number_input("COTE DU BOOKMAKER", 1.01, 100.0, 2.0)
        st.markdown("</div>", unsafe_allow_html=True)

        # --- 4. AUDIT DE MISE ---
        st.markdown("<div class='audit-gold'><h3>🛡️ 3. AUDIT & MISE CALCULÉE</h3>", unsafe_allow_html=True)
        p_map = {"Victoire Domicile": r['Home'], "Match Nul": r['Draw'], "Victoire Extérieur": r['Away'], "1N": r['1N'], "N2": r['N2'], "BTTS OUI": r['BTTS_Yes']}
        prob = p_map[type_pari]
        ev = prob * cote_book
        
        if ev > 1.02:
            # Calcul Kelly pur : (p*b - q) / b
            b = cote_book - 1
            kelly_pur = (prob * b - (1 - prob)) / b
            
            if kelly_pur > 0:
                mise_finale = round(bankroll * kelly_pur * k_frac, 2)
                st.markdown(f"✅ **RENTABILITÉ DÉTECTÉE (EV: {ev:.2f})**")
                st.markdown(f"Indice de confiance : {(prob**2*cote_book):.2f}")
                st.markdown(f"<p class='stake-val'>{mise_finale} €</p>", unsafe_allow_html=True)
                st.write(f"Conseil : Placez {mise_finale}€ sur ce pari selon votre profil {mode_mise}.")
            else:
                st.error("L'avantage est trop faible pour suggérer une mise Kelly.")
        else:
            st.error(f"⚠️ PARI NON RENTABLE (EV: {ev:.2f}). L'algorithme déconseille de jouer.")
        st.markdown("</div>", unsafe_allow_html=True)

with tab2:
    st.subheader("📡 GÉNÉRATEUR AUTOMATIQUE")
    # Le générateur suit la même logique : il liste les opportunités et calcule 
    # la mise en fonction de k_frac choisi en sidebar.
    st.info("Sélectionnez votre plage de dates et vos ligues. Le ticket généré inclura le montant de mise suggéré pour chaque ligne.")
