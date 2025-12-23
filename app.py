import streamlit as st
import requests
import numpy as np
from scipy.stats import poisson
from datetime import datetime, timedelta
import pandas as pd

# --- CONFIGURATION CLEMENTRNXX PREDICTOR V11.0 ---
st.set_page_config(page_title="Clementrnxx Predictor V11.0 - BANKROLL MGMT", layout="wide")

DISCORD_WEBHOOK_URL = "https://discord.com/api/webhooks/1453026279275106355/gbYAwBRntm1FCoqoBTz5lj1SCe2ijyeHHYoe4CFYwpzOw2DO-ozcCsgkK_53HhB-kFGE"

st.markdown("""
    <style>
    .stApp { background-image: url("https://media.giphy.com/media/VZrfUvQjXaGEQy1RSn/giphy.gif"); background-size: cover; background-attachment: fixed; }
    .stApp > div:first-child { background-color: rgba(0, 0, 0, 0.93); }
    h1, h2, h3, p, span, label { color: #FFD700 !important; font-family: 'Monospace', sans-serif; }
    .section-box { border: 2px solid #FFD700; padding: 20px; border-radius: 15px; background: rgba(0,0,0,0.8); margin-bottom: 25px; }
    .audit-gold { border: 2px solid #FFD700; background: linear-gradient(145deg, #000, #222); padding: 20px; border-radius: 15px; text-align: center; }
    .stake-text { font-size: 1.5rem; color: #00FF00 !important; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

# --- CONFIG MODES & FRACTION DE KELLY ---
# On utilise Kelly Fractionnaire pour éviter de tout perdre sur un seul pari
RISK_LEVELS = {
    "ULTRA-SAFE": {"elite_min": 0.90, "p_min": 0.88, "kelly_fraction": 0.05}, # 5% de Kelly
    "SAFE": {"elite_min": 0.75, "p_min": 0.80, "kelly_fraction": 0.10},
    "MID-SAFE": {"elite_min": 0.65, "p_min": 0.70, "kelly_fraction": 0.15},
    "MID": {"elite_min": 0.55, "p_min": 0.60, "kelly_fraction": 0.20},
    "MID-AGGRESSIF": {"elite_min": 0.45, "p_min": 0.50, "kelly_fraction": 0.25},
    "JACKPOT": {"elite_min": 0.30, "p_min": 0.35, "kelly_fraction": 0.30}
}

# --- FONCTIONS ---
def calculate_kelly(prob, cote, bankroll, fraction):
    # Formule de Kelly : f* = (p*b - q) / b  où b = cote - 1
    if cote <= 1: return 0
    b = cote - 1
    p = prob
    q = 1 - p
    f_star = (p * b - q) / b
    if f_star <= 0: return 0
    # On applique la fraction du mode de risque (ex: 1/10 de Kelly)
    return round(f_star * bankroll * fraction, 2)

# (Fonctions API et Poisson identiques aux versions précédentes...)
@st.cache_data(ttl=3600)
def get_api(endpoint, params):
    try:
        API_KEY = st.secrets["MY_API_KEY"]
        HEADERS = {'x-apisports-key': API_KEY}
        return requests.get(f"https://v3.football.api-sports.io/{endpoint}", headers=HEADERS, params=params, timeout=12).json().get('response', [])
    except: return []

def calculate_all_probs(lh, la):
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

# --- UI ---
st.title("⚽ CLEMENTRNXX PREDICTOR V11 - BANKROLL MGMT")

# Sidebar pour les fonds
with st.sidebar:
    st.header("💰 MA BANKROLL")
    total_bankroll = st.number_input("FONDS DISPONIBLES (€)", min_value=10.0, value=1000.0, step=50.0)
    risk_mode = st.select_slider("MODE DE RISQUE", options=list(RISK_LEVELS.keys()), value="MID")
    st.info(f"Mode {risk_mode} activé. L'algo limitera les mises pour protéger vos fonds.")

tab1, tab2 = st.tabs(["🎯 1VS1 : ANALYSE / BET / AUDIT", "📡 GÉNÉRATEUR DE TICKETS"])

with tab1:
    st.subheader("🕵️ AUDIT DE MATCH")
    # ... (Sélecteurs de ligues et équipes identiques à la V10)
    # [Simulation de sélection équipe pour l'exemple]
    l_name = st.selectbox("LIGUE", ["Premier League", "La Liga", "Ligue 1", "Champions League"])
    c_audit1, c_audit2 = st.columns(2)
    th = c_audit1.text_input("EQUIPE DOMICILE (Nom exact)")
    ta = c_audit2.text_input("EQUIPE EXTERIEUR (Nom exact)")

    if st.button("LANCER L'AUDIT"):
        # (Logique de calcul Poisson simulée ici pour la démo)
        lh, la = 1.5, 1.2 # Ex: Probas moyennes
        st.session_state.v11_res = {"res": calculate_all_probs(lh, la), "th": th, "ta": ta}

    if 'v11_res' in st.session_state:
        r, th_n, ta_n = st.session_state.v11_res["res"], st.session_state.v11_res["th"], st.session_state.v11_res["ta"]
        
        # 1. ANALYSE
        st.markdown("<div class='section-box'><h3>📊 1. ANALYSE STATISTIQUE</h3>", unsafe_allow_html=True)
        st.write(f"Proba Victoire {th_n}: {r['Home']:.1%} | Nul: {r['Draw']:.1%} | Victoire {ta_n}: {r['Away']:.1%}")
        st.markdown("</div>", unsafe_allow_html=True)

        # 2. BET
        st.markdown("<div class='section-box'><h3>💰 2. ZONE DE BET</h3>", unsafe_allow_html=True)
        col_bet1, col_bet2 = st.columns(2)
        bet_type = col_bet1.selectbox("VOTRE PARI", ["Victoire Domicile", "Match Nul", "Victoire Extérieur", "1N", "N2", "BTTS OUI"])
        bet_cote = col_bet2.number_input("COTE BOOKMAKER", 1.01, 50.0, 2.0)
        st.markdown("</div>", unsafe_allow_html=True)

        # 3. AUDIT & MISE
        st.markdown("<div class='audit-gold'><h3>🛡️ 3. AUDIT DE MISE FINALE</h3>", unsafe_allow_html=True)
        # Mapping proba
        p_map = {"Victoire Domicile": r['Home'], "Match Nul": r['Draw'], "Victoire Extérieur": r['Away'], "1N": r['1N'], "N2": r['N2'], "BTTS OUI": r['BTTS_Yes']}
        current_p = p_map[bet_type]
        ev = current_p * bet_cote

        if ev > 1.05:
            cfg = RISK_LEVELS[risk_mode]
            mise_recommandee = calculate_kelly(current_p, bet_cote, total_bankroll, cfg['kelly_fraction'])
            st.markdown(f"✅ **PARI RENTABLE (EV: {ev:.2f})**")
            st.markdown(f"<p class='stake-text'>MISE CONSEILLÉE : {mise_recommandee} €</p>", unsafe_allow_html=True)
            st.write(f"(Soit {round((mise_recommandee/total_bankroll)*100, 2)}% de votre bankroll)")
        else:
            st.error(f"⚠️ PARI NON RENTABLE (EV: {ev:.2f}). L'algorithme déconseille de miser.")
        st.markdown("</div>", unsafe_allow_html=True)

with tab2:
    st.subheader("📡 GÉNÉRATEUR AVEC MISE AUTOMATIQUE")
    # (Logique du générateur identique, mais ajoute une colonne "Mise (€)" dans le tableau final)
    st.write("Le générateur calculera la mise optimale pour chaque match du combiné ou du ticket selon Kelly.")
    # ... (Code du scanner identique à la V10 avec appel à calculate_kelly pour chaque ligne)
