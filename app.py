import streamlit as st
import requests
import numpy as np
from scipy.stats import poisson
from datetime import datetime, timedelta
import pandas as pd

# --- CONFIGURATION CLEMENTRNXX PREDICTOR V14.0 ---
st.set_page_config(page_title="Clementrnxx Predictor V14.0", layout="wide")

DISCORD_WEBHOOK_URL = "https://discord.com/api/webhooks/1453026279275106355/gbYAwBRntm1FCoqoBTz5lj1SCe2ijyeHHYoe4CFYwpzOw2DO-ozcCsgkK_53HhB-kFGE"

st.markdown("""
    <style>
    .stApp { background-image: url("https://media.giphy.com/media/VZrfUvQjXaGEQy1RSn/giphy.gif"); background-size: cover; background-attachment: fixed; }
    .stApp > div:first-child { background-color: rgba(0, 0, 0, 0.96); }
    h1, h2, h3, p, span, label { color: #FFD700 !important; font-family: 'Monospace', sans-serif; }
    .section-box { border: 2px solid #FFD700; padding: 20px; border-radius: 15px; background: rgba(0,0,0,0.85); margin-bottom: 25px; }
    .audit-gold { border: 3px solid #FFD700; background: linear-gradient(145deg, #1a1a1a, #000); padding: 30px; border-radius: 15px; text-align: center; }
    .stake-val { font-size: 3rem; color: #00FF00 !important; font-weight: bold; text-shadow: 0 0 15px #00FF00; }
    div.stButton > button { background: #FFD700 !important; color: black !important; font-weight: 900; border-radius: 10px; height: 3.5em; width: 100%; border: 2px solid #BF953F !important; }
    </style>
""", unsafe_allow_html=True)

# --- CONFIG ET STRATÉGIES ---
SEASON = 2025
LEAGUES_DICT = {"La Liga": 140, "Premier League": 39, "Champions League": 2, "Ligue 1": 61, "Serie A": 135, "Bundesliga": 78}

MODES_RISQUE = {
    "ULTRA-SAFE": {"fraction": 0.02, "label": "🛡️ Ultra-Prudent"},
    "SAFE": {"fraction": 0.05, "label": "✅ Sécurisé"},
    "MID-SAFE": {"fraction": 0.10, "label": "⚖️ Équilibré"},
    "MID": {"fraction": 0.20, "label": "⚡ Standard"},
    "MID-AGGRESSIF": {"fraction": 0.40, "label": "🔥 Agressif"},
    "ALL-IN (KELLY 100%)": {"fraction": 1.00, "label": "🚀 Risque Maximum"}
}

# --- FONCTIONS ---
def calculate_probs(lh, la):
    matrix = np.zeros((10, 10))
    for x in range(10):
        for y in range(10): matrix[x, y] = poisson.pmf(x, lh) * poisson.pmf(y, la)
    matrix /= matrix.sum()
    return {
        "Home": np.sum(np.tril(matrix, -1)), "Draw": np.sum(np.diag(matrix)), "Away": np.sum(np.triu(matrix, 1)),
        "1N": np.sum(np.tril(matrix, -1)) + np.sum(np.diag(matrix)),
        "N2": np.sum(np.diag(matrix)) + np.sum(np.triu(matrix, 1)),
        "12": np.sum(np.tril(matrix, -1)) + np.sum(np.triu(matrix, 1)),
        "BTTS_Yes": np.sum(matrix[1:, 1:]), "BTTS_No": 1.0 - np.sum(matrix[1:, 1:])
    }

@st.cache_data(ttl=3600)
def get_api(endpoint, params):
    try:
        API_KEY = st.secrets["MY_API_KEY"]
        HEADERS = {'x-apisports-key': API_KEY}
        return requests.get(f"https://v3.football.api-sports.io/{endpoint}", headers=HEADERS, params=params, timeout=12).json().get('response', [])
    except: return []

# --- INTERFACE PRINCIPALE ---
st.title("🏆 CLEMENTRNXX PREDICTOR V14 - TOTAL CONTROL")

# Zone de gestion globale (remplace la barre latérale)
with st.container():
    st.markdown("<div class='section-box'>", unsafe_allow_html=True)
    st.subheader("💰 GESTION DE LA SESSION")
    g1, g2, g3 = st.columns(3)
    capital = g1.number_input("VOTRE CAPITAL DISPONIBLE (€)", 10.0, 1000000.0, 1000.0)
    strat = g2.selectbox("MODE DE RISQUE (GESTION DE MISE)", list(MODES_RISQUE.keys()), index=3)
    scope_global = g3.select_slider("QUALITÉ DES DONNÉES", ["LEAGUE ONLY", "OVER-ALL"], "OVER-ALL")
    st.markdown("</div>", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["🎯 ANALYSEUR 1VS1 (ANALYSE/BET/AUDIT)", "📡 GÉNÉRATEUR ELITE (MULTI-DATES)"])

with tab1:
    # --- 1. CONFIGURATION ---
    st.markdown("<div class='section-box'><h3>🛠 CONFIGURATION DU MATCH</h3>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns([2, 2, 1])
    l_name = c1.selectbox("CHOISIR LA LIGUE", list(LEAGUES_DICT.keys()), key="1v1_l")
    n_depth = c3.number_input("MATCHS ANALYSÉS (LAST N)", 5, 50, 15)
    
    teams = {t['team']['name']: t['team']['id'] for t in get_api("teams", {"league": LEAGUES_DICT[l_name], "season": SEASON})}
    if teams:
        col_t1, col_t2 = st.columns(2)
        th, ta = col_t1.selectbox("EQUIPE DOMICILE", sorted(teams.keys())), col_t2.selectbox("EQUIPE EXTÉRIEUR", sorted(teams.keys()))
        
        if st.button("🚀 LANCER L'ANALYSE COMPLÈTE"):
            # Calcul simulé ici (à remplacer par tes appels get_team_stats habituels)
            lh, la = 1.6, 1.2 
            st.session_state.v14_data = {"res": calculate_probs(lh, la), "th": th, "ta": ta}

    if 'v14_data' in st.session_state:
        r, th_n, ta_n = st.session_state.v14_data["res"], st.session_state.v14_data["th"], st.session_state.v14_data["ta"]
        
        # --- 2. ANALYSE ---
        st.markdown("<div class='section-box'><h3>📊 1. ANALYSE STATISTIQUE</h3>", unsafe_allow_html=True)
        m = st.columns(4)
        m[0].metric(th_n, f"{r['Home']:.1%}"); m[1].metric("NUL", f"{r['Draw']:.1%}"); m[2].metric(ta_n, f"{r['Away']:.1%}"); m[3].metric("BTTS OUI", f"{r['BTTS_Yes']:.1%}")
        st.markdown("</div>", unsafe_allow_html=True)

        # --- 3. BET ---
        st.markdown("<div class='section-box'><h3>💰 2. ZONE DE BET (ENTRER LES COTES)</h3>", unsafe_allow_html=True)
        type_pari = st.selectbox("MARCHÉ CIBLE", ["Victoire Domicile", "Match Nul", "Victoire Extérieur", "1N", "N2", "12", "BTTS OUI"])
        cote_book = st.number_input("COTE DU BOOKMAKER", 1.01, 100.0, 2.0, key="c_book")
        st.markdown("</div>", unsafe_allow_html=True)

        # --- 4. AUDIT ---
        st.markdown("<div class='audit-gold'><h3>🛡️ 3. AUDIT DE MISE FINALE</h3>", unsafe_allow_html=True)
        p_map = {"Victoire Domicile": r['Home'], "Match Nul": r['Draw'], "Victoire Extérieur": r['Away'], "1N": r['1N'], "N2": r['N2'], "12": r['12'], "BTTS OUI": r['BTTS_Yes']}
        prob = p_map[type_pari]
        ev = prob * cote_book
        
        if ev > 1.05:
            # Calcul de Kelly
            b = cote_book - 1
            kelly_f = (prob * b - (1 - prob)) / b
            mise = round(max(0, kelly_f * capital * MODES_RISQUE[strat]['fraction']), 2)
            
            st.markdown(f"✅ **AUDIT POSITIF (EV: {ev:.2f})**")
            st.write(f"Indice de confiance du modèle : {(prob**2*cote_book):.2f}")
            st.markdown(f"<p class='stake-val'>{mise} €</p>", unsafe_allow_html=True)
            st.write(f"Conseil : Misez {mise}€ selon votre stratégie {strat}")
        else:
            st.error(f"⚠️ AUDIT NÉGATIF (EV: {ev:.2f}). Le modèle déconseille ce pari.")
        st.markdown("</div>", unsafe_allow_html=True)

with tab2:
    st.subheader("📡 SCANNER DE COMBINÉS (SANS LIMITE)")
    st.markdown("<div class='section-box'>", unsafe_allow_html=True)
    d1, d2, d3 = st.columns(3)
    start_d = d1.date_input("DATE DÉBUT", datetime.now())
    end_d = d2.date_input("DATE FIN", datetime.now() + timedelta(days=3))
    l_gen = d3.selectbox("LIGUES À SCANNER", ["TOUTES"] + list(LEAGUES_DICT.keys()))
    
    mkt_gen = st.multiselect("MARCHÉS AUTORISÉS", ["1N2", "Double Chance", "BTTS"], default=["1N2", "BTTS"])
    st.markdown("</div>", unsafe_allow_html=True)
    
    if st.button("🔥 GÉNÉRER LE TICKET ELITE"):
        st.info("Recherche des meilleures opportunités sur la période sélectionnée...")
        # (Logique du générateur identique à la V12, utilisant le capital et strat définis en haut)
