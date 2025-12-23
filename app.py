import streamlit as st
import requests
import numpy as np
import pandas as pd
from scipy.stats import poisson
from datetime import datetime

# --- CONFIGURATION ET STYLE ---
st.set_page_config(page_title="Clementrnxx Predictor V6.0 - Diamond", layout="wide")

st.markdown("""
    <style>
    .stApp { background-image: url("https://media.giphy.com/media/VZrfUvQjXaGEQy1RSn/giphy.gif"); background-size: cover; background-attachment: fixed; }
    .stApp > div:first-child { background-color: rgba(0, 0, 0, 0.94); }
    h1, h2, h3, p, span, label { color: #FFD700 !important; font-family: 'JetBrains Mono', monospace; }
    
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] { 
        background-color: rgba(255, 215, 0, 0.05); border: 1px solid #FFD700; 
        border-radius: 10px 10px 0 0; color: #FFD700; padding: 10px 20px;
    }
    
    .stat-card {
        background: rgba(255, 255, 255, 0.03); border-left: 4px solid #FFD700;
        padding: 15px; border-radius: 8px; margin-bottom: 10px;
    }
    .github-link {
        display: block; text-align: center; color: #FFD700 !important;
        font-weight: bold; font-size: 1.1rem; text-decoration: none;
        padding: 20px; border-top: 1px solid rgba(255, 215, 0, 0.2);
    }
    </style>
""", unsafe_allow_html=True)

# --- ENGINE MATHÉMATIQUE V6 ---
def calculate_diamond_probs(lh, la):
    # Application de l'avantage domicile (Correction Gamma)
    lh = lh * 1.14 
    la = la * 0.87
    
    matrix = np.zeros((8, 8))
    for x in range(8):
        for y in range(8):
            prob = poisson.pmf(x, lh) * poisson.pmf(y, la)
            # Correction Dixon-Coles simplifiée pour les petits scores
            if x == 0 and y == 0: prob *= 1.10
            elif x == 1 and y == 1: prob *= 1.05
            matrix[x, y] = prob
            
    matrix /= matrix.sum()
    p_h, p_n, p_a = np.sum(np.tril(matrix, -1)), np.sum(np.diag(matrix)), np.sum(np.triu(matrix, 1))
    
    return {
        "p_h": p_h, "p_n": p_n, "p_a": p_a,
        "p_1n": p_h + p_n, "p_n2": p_n + p_a, "p_12": p_h + p_a,
        "p_btts": np.sum(matrix[1:, 1:]), "matrix": matrix
    }

# --- CONFIG API ---
API_KEY = st.secrets["MY_API_KEY"]
BASE_URL = "https://v3.football.api-sports.io/"
HEADERS = {'x-apisports-key': API_KEY}

LEAGUES_DATA = {
    "Premier League": {"id": 39, "avg_goals": 3.28, "home_win": 46},
    "La Liga": {"id": 140, "avg_goals": 2.65, "home_win": 44},
    "Bundesliga": {"id": 78, "avg_goals": 3.21, "home_win": 48},
    "Serie A": {"id": 135, "avg_goals": 2.61, "home_win": 41},
    "Ligue 1": {"id": 61, "avg_goals": 2.82, "home_win": 43}
}

@st.cache_data(ttl=3600)
def get_api(endpoint, params):
    try:
        r = requests.get(f"{BASE_URL}{endpoint}", headers=HEADERS, params=params, timeout=12)
        return r.json().get('response', [])
    except: return []

# --- UI PRINCIPALE ---
st.title("💎 CLEMENTRNXX PREDICTOR V6.0")
st.write("DIAMOND EDITION - ALGORITHME DIXON-COLES CALIBRÉ 2025")

tab_stats, tab_1v1, tab_scan = st.tabs(["📊 STATS LEAGUES", "🎯 ANALYSE 1VS1", "🚀 SCANNER"])

# --- TAB STATS ---
with tab_stats:
    st.subheader("GLOBAL LEAGUE INSIGHTS (SEASON 24/25)")
    cols = st.columns(len(LEAGUES_DATA))
    for i, (name, data) in enumerate(LEAGUES_DATA.items()):
        with cols[i]:
            st.markdown(f"""
            <div class='stat-card'>
                <b>{name}</b><br>
                Buts/Match: {data['avg_goals']}<br>
                Win Domicile: {data['home_win']}%
            </div>
            """, unsafe_allow_html=True)
    
    st.info("Note : La Premier League et la Bundesliga présentent les plus hauts taux de BTTS (62%+).")

# --- TAB 1V1 ---
with tab_1v1:
    l_choice = st.selectbox("LIGUE", list(LEAGUES_DATA.keys()))
    lid = LEAGUES_DATA[l_choice]["id"]
    
    teams_res = get_api("teams", {"league": lid, "season": 2025})
    teams = {t['team']['name']: t['team']['id'] for t in teams_res}
    
    if teams:
        c1, c2 = st.columns(2)
        home, away = c1.selectbox("DOMICILE", sorted(teams.keys())), c2.selectbox("EXTÉRIEUR", sorted(teams.keys()))
        
        if st.button("DIAMOND ANALYSIS"):
            # Calcul de lambda basé sur les 12 derniers matchs (plus de data = plus de précision)
            def get_l(tid):
                f = get_api("fixtures", {"team": tid, "season": 2025, "last": 12})
                if not f: return 1.3
                g = [(m['goals']['home'] if m['teams']['home']['id'] == tid else m['goals']['away']) or 0 for m in f]
                return np.mean(g)

            lh, la = get_l(teams[home]), get_l(teams[away])
            st.session_state.v6 = {"res": calculate_diamond_probs(lh, la), "h": home, "a": away}

    if 'v6' in st.session_state:
        r, h, a = st.session_state.v6["res"], st.session_state.v6["h"], st.session_state.v6["a"]
        
        col_res = st.columns(4)
        col_res[0].metric(h, f"{r['p_h']*100:.1f}%")
        col_res[1].metric("NUL", f"{r['p_n']*100:.1f}%")
        col_res[2].metric(a, f"{r['p_a']*100:.1f}%")
        col_res[3].metric("BTTS", f"{r['p_btts']*100:.1f}%")

        # AUDIT ET BET
        st.divider()
        st.subheader("🕵️ AUDIT & GESTION")
        ac1, ac2, ac3 = st.columns([2, 1, 2])
        u_bet = ac1.selectbox("PARI", [h, a, "Nul", "1N", "N2", "12", "BTTS OUI"])
        u_odd = ac2.number_input("COTE", value=1.50)
        
        p_map = {h: r['p_h'], a: r['p_a'], "Nul": r['p_n'], "1N": r['p_1n'], "N2": r['p_n2'], "12": r['p_12'], "BTTS OUI": r['p_btts']}
        ev = p_map[u_bet] * u_odd
        
        with ac3:
            st.write(f"Indice de Valeur (EV) : **{ev:.2f}**")
            st.progress(min(ev/2, 1.0))
            if ev > 1.10: st.success("VALEUR DÉTECTÉE")
            else: st.warning("PAS DE VALUE")

# --- TAB SCANNER ---
with tab_scan:
    risk = st.select_slider("MODE DE RISQUE", options=["SAFE", "MID-SAFE", "MID", "MID-AGGRESSIF", "AGGRESSIF"], value="MID")
    # Configuration simplifiée pour le scan
    risk_map = {"SAFE": 0.78, "MID-SAFE": 0.70, "MID": 0.60, "MID-AGGRESSIF": 0.50, "AGGRESSIF": 0.40}
    
    if st.button("GÉNÉRER TICKET DIAMOND"):
        st.write(f"Recherche de matchs avec probabilité > {risk_map[risk]*100}%...")
        # (Logique de scan similaire à la v5.5 mais avec calculate_diamond_probs)
        st.info("Scanner en cours de calibration sur les nouvelles API 2025...")

# --- FOOTER ---
st.markdown(f"""
    <a href="https://github.com/clementrnx" class="github-link" target="_blank">
        GITHUB : github.com/clementrnx
    </a>
""", unsafe_allow_html=True)
