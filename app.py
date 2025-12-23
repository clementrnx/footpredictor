import streamlit as st
import requests
import numpy as np
import pandas as pd
from scipy.stats import poisson

# --- CONFIGURATION PAGE ---
st.set_page_config(page_title="iTrOz Predictor Master", layout="wide")

# --- STYLE CSS (JAUNE ET NOIR, GLASSMORPHISM, GIF BACKGROUND) ---
st.markdown("""
    <style>
    .stApp {
        background-image: url("https://media.giphy.com/media/VZrfUvQjXaGEQy1RSn/giphy.gif");
        background-size: cover;
        background-attachment: fixed;
    }
    .stApp > div:first-child { background-color: rgba(0, 0, 0, 0.94); }
    
    h1, h2, h3, p, span, label, .stMetric div { 
        color: #FFD700 !important; 
        font-family: 'Monospace', sans-serif !important; 
        text-transform: uppercase;
    }

    .glass-card {
        background: rgba(255, 255, 255, 0.02) !important;
        backdrop-filter: blur(15px);
        border: 1px solid rgba(255, 215, 0, 0.2);
        border-radius: 15px;
        padding: 25px;
        margin-bottom: 25px;
    }

    div[data-baseweb="select"], div[data-baseweb="input"], .stNumberInput input, div[data-baseweb="radio"] {
        background-color: rgba(0, 0, 0, 0.8) !important;
        border: 1px solid rgba(255, 215, 0, 0.3) !important;
        color: #FFD700 !important;
    }

    div.stButton > button {
        background: rgba(255, 215, 0, 0.05) !important;
        border: 1px solid #FFD700 !important;
        color: #FFD700 !important;
        height: 3.5em;
        font-weight: bold;
        width: 100%;
    }
    
    div.stButton > button:hover { 
        background: #FFD700 !important; 
        color: #000 !important;
    }
    </style>
""", unsafe_allow_html=True)

# --- LOGIQUE API ---
API_KEY = st.secrets.get("MY_API_KEY", "")
HEADERS = {'x-apisports-key': API_KEY}
BASE_URL = "https://v3.football.api-sports.io/"
SEASON = 2025

@st.cache_data(ttl=3600)
def call_api(endpoint, params):
    try:
        r = requests.get(f"{BASE_URL}{endpoint}", headers=HEADERS, params=params, timeout=12)
        return r.json().get('response', [])
    except: return []

def get_stats(team_id, league_id):
    f = call_api("fixtures", {"team": team_id, "league": league_id, "season": SEASON, "last": 10})
    if not f: return 1.3, 0.5
    scored = [float(m['teams']['home' if m['teams']['home']['id'] == team_id else 'away'].get('xg') or m['goals']['home' if m['teams']['home']['id'] == team_id else 'away'] or 0) for m in f]
    return np.mean(scored), np.std(scored)

# --- INITIALISATION SESSION STATE (Pour éviter le reset) ---
if 'results' not in st.session_state:
    st.session_state.results = None

# --- INTERFACE ---
st.markdown("<h1 style='text-align:center; letter-spacing:15px; margin-bottom:40px;'>ITROZ PREDICTOR</h1>", unsafe_allow_html=True)

# 1. PARAMETRES
st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
c_bank, c_risk = st.columns([1, 2])
bankroll = c_bank.number_input("CAPITAL (€)", value=100.0)
risk_mode = c_risk.radio("MODE DE MISE", ["🛡️ SAFE", "⚖️ MID", "🔥 JOUEUR"], horizontal=True)
risk_map = {"🛡️ SAFE": 0.1, "⚖️ MID": 0.25, "🔥 JOUEUR": 0.5}
st.markdown("</div>", unsafe_allow_html=True)

# 2. SELECTION
st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
leagues = {"Premier League": 39, "Ligue 1": 61, "La Liga": 140, "Bundesliga": 78, "Serie A": 135}
col_l, col_h, col_a = st.columns(3)
l_name = col_l.selectbox("CHAMPIONNAT", list(leagues.keys()))
teams_data = call_api("teams", {"league": leagues[l_name], "season": SEASON})
team_map = {t['team']['name']: t['team']['id'] for t in teams_data}
t_h = col_h.selectbox("DOMICILE", sorted(team_map.keys()))
t_a = col_a.selectbox("EXTERIEUR", sorted(team_map.keys()), index=1)

if st.button("LANCER L'ANALYSE"):
    with st.spinner("CALCUL..."):
        m_h, s_h = get_stats(team_map[t_h], leagues[l_name])
        m_a, s_a = get_stats(team_map[t_a], leagues[l_name])
        lh, la = m_h * (m_a/1.3), m_a * (m_h/1.3)
        matrix = np.outer(poisson.pmf(np.arange(6), lh), poisson.pmf(np.arange(6), la))
        matrix /= matrix.sum()
        
        st.session_state.results = {
            "p_h": np.sum(np.tril(matrix, -1)),
            "p_n": np.sum(np.diag(matrix)),
            "p_a": np.sum(np.triu(matrix, 1)),
            "p_btts": np.sum(matrix[1:, 1:]),
            "p_o25": sum(matrix[i, j] for i in range(6) for j in range(6) if i+j > 2.5),
            "matrix": matrix,
            "t_h": t_h,
            "t_a": t_a
        }
st.markdown("</div>", unsafe_allow_html=True)

# AFFICHAGE DES RESULTATS (Si ils existent en mémoire)
if st.session_state.results:
    res = st.session_state.results
    
    # 3. PROBAS
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    r1, r2, r3 = st.columns(3)
    r1.metric(res['t_h'], f"{res['p_h']*100:.1f}%")
    r2.metric("NUL", f"{res['p_n']*100:.1f}%")
    r3.metric(res['t_a'], f"{res['p_a']*100:.1f}%")
    st.markdown("</div>", unsafe_allow_html=True)

    # 4. AUDIT ET COTES
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown(f"### ANALYSE ET AUDIT : {res['t_h']} VS {res['t_a']}")
    
    col_audit_1, col_audit_2 = st.columns(2)
    
    with col_audit_1:
        type_bet = st.selectbox("AUDITER UN PRONO", [
            f"VICTOIRE : {res['t_h']}", 
            "MATCH NUL", 
            f"VICTOIRE : {res['t_a']}", 
            "LES DEUX MARQUENT (BTTS)", 
            "PLUS DE 2.5 BUTS"
        ])
        cote_jouee = st.number_input("COTE DU BOOKMAKER", value=1.80, step=0.05)
        
        audit_map = {
            f"VICTOIRE : {res['t_h']}": res['p_h'],
            "MATCH NUL": res['p_n'],
            f"VICTOIRE : {res['t_a']}": res['p_a'],
            "LES DEUX MARQUENT (BTTS)": res['p_btts'],
            "PLUS DE 2.5 BUTS": res['p_o25']
        }
        
        prob_target = audit_map[type_bet]
        ev = prob_target * cote_jouee
        
        if ev >= 1.05:
            st.success(f"VALIDE - VALUE DETECTEE : {ev:.2f}")
            k = ((cote_jouee-1)*prob_target - (1-prob_target)) / (cote_jouee-1)
            mise = max(0, bankroll * k * risk_map[risk_mode])
            st.write(f"MISE CONSEILLEE : **{mise:.2f}€**")
        else:
            st.error(f"DECONSEILLE - RENTABILITE INSUFFISANTE : {ev:.2f}")

    with col_audit_2:
        st.markdown("### SCORES LES PLUS PROBABLES")
        flat = res['matrix'].flatten()
        top3 = np.argsort(flat)[-3:][::-1]
        for idx in top3:
            h, a = divmod(idx, 6)
            st.write(f"SCORE {h}-{a} : {res['matrix'][h,a]*100:.1f}%")
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<p style='text-align:center; opacity:0.3; letter-spacing:5px;'>ITROZ SYSTEM v9.0</p>", unsafe_allow_html=True)
