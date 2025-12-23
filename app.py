import streamlit as st
import requests
import numpy as np
import pandas as pd
from scipy.stats import poisson

# --- CONFIGURATION PAGE ---
st.set_page_config(page_title="iTrOz Predictor Master", layout="wide")

# --- STYLE CSS (JAUNE, NOIR, GLASSMORPHISM, GIF BACKGROUND) ---
st.markdown("""
    <style>
    @keyframes subtleDistort {
        0% { filter: hue-rotate(0deg) brightness(1); }
        50% { filter: hue-rotate(2deg) brightness(1.1); }
        100% { filter: hue-rotate(0deg) brightness(1); }
    }

    .stApp {
        background-image: url("https://media.giphy.com/media/VZrfUvQjXaGEQy1RSn/giphy.gif");
        background-size: cover;
        background-attachment: fixed;
    }

    /* Overlay sombre pour lisibilité */
    .stApp > div:first-child { 
        background-color: rgba(0, 0, 0, 0.9); 
    }
    
    /* Global Text */
    h1, h2, h3, p, span, label, .stMetric div { 
        color: #FFD700 !important; 
        font-family: 'Monospace', sans-serif !important; 
        letter-spacing: 1px;
    }

    /* Glassmorphism Containers */
    .glass-card {
        background: rgba(255, 255, 255, 0.03) !important;
        backdrop-filter: blur(15px) !important;
        -webkit-backdrop-filter: blur(15px) !important;
        border: 1px solid rgba(255, 215, 0, 0.2) !important;
        border-radius: 15px !important;
        padding: 25px;
        margin-bottom: 20px;
    }

    /* Inputs & Selectbox */
    div[data-baseweb="select"], div[data-baseweb="input"], .stNumberInput input {
        background-color: rgba(0, 0, 0, 0.6) !important;
        border: 1px solid rgba(255, 215, 0, 0.3) !important;
        color: #FFD700 !important;
    }

    /* Styled Buttons */
    div.stButton > button {
        background: rgba(255, 215, 0, 0.1) !important;
        border: 1px solid #FFD700 !important;
        color: #FFD700 !important;
        border-radius: 10px !important;
        text-transform: uppercase !important;
        font-weight: bold !important;
        transition: 0.4s all;
        width: 100%;
    }
    
    div.stButton > button:hover { 
        background: #FFD700 !important; 
        color: #000 !important;
        box-shadow: 0 0 20px rgba(255, 215, 0, 0.4);
    }

    /* Table & Dataframe Customization */
    .stDataFrame, [data-testid="stTable"] {
        background: rgba(0,0,0,0.5) !important;
        border: 1px solid rgba(255, 215, 0, 0.2);
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

# --- INTERFACE ---
st.markdown("<h1 style='text-align:center; letter-spacing:10px;'>ITROZ PREDICTOR</h1>", unsafe_allow_html=True)

# Barre Latérale / Settings
with st.sidebar:
    st.markdown("### CONFIGURATION")
    bankroll = st.number_input("CAPITAL TOTAL", value=100.0)
    st.markdown("### MODE DE PARI")
    risk_mode = st.radio("PROFIL", ["🛡️ SAFE", "⚖️ MID", "🔥 JOUEUR"], label_visibility="collapsed")
    risk_map = {"🛡️ SAFE": 0.1, "⚖️ MID": 0.25, "🔥 JOUEUR": 0.5}

# Sélection Match
leagues = {"Premier League": 39, "Ligue 1": 61, "La Liga": 140, "Bundesliga": 78, "Serie A": 135}
st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
col_l, col_h, col_a = st.columns(3)
l_name = col_l.selectbox("LIGUE", list(leagues.keys()))
league_id = leagues[l_name]

teams_data = call_api("teams", {"league": league_id, "season": SEASON})
team_map = {t['team']['name']: t['team']['id'] for t in teams_data}

t_h = col_h.selectbox("DOMICILE", sorted(team_map.keys()))
t_a = col_a.selectbox("EXTERIEUR", sorted(team_map.keys()), index=1)
st.markdown("</div>", unsafe_allow_html=True)

if st.button("LANCER L'ANALYSE"):
    with st.spinner("CALCUL EN COURS..."):
        # Stats & Modélisation
        m_h, s_h = get_stats(team_map[t_h], league_id)
        m_a, s_a = get_stats(team_map[t_a], league_id)
        
        lh, la = m_h * (m_a/1.3), m_a * (m_h/1.3)
        matrix = np.outer(poisson.pmf(np.arange(6), lh), poisson.pmf(np.arange(6), la))
        matrix /= matrix.sum()
        
        p_h, p_n, p_a = np.sum(np.tril(matrix, -1)), np.sum(np.diag(matrix)), np.sum(np.triu(matrix, 1))

        # Résultats Principaux
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        res1, res2, res3 = st.columns(3)
        res1.metric(t_h.upper(), f"{p_h*100:.1f}%")
        res2.metric("NUL", f"{p_n*100:.1f}%")
        res3.metric(t_a.upper(), f"{p_a*100:.1f}%")
        st.markdown("</div>", unsafe_allow_html=True)

        # Value & Kelly
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.subheader("VALUE ET MISE")
        vc1, vc2, vc3 = st.columns(3)
        c_h = vc1.number_input(f"COTE {t_h}", value=2.0)
        c_n = vc2.number_input("COTE NUL", value=3.2)
        c_a = vc3.number_input(f"COTE {t_a}", value=3.5)
        
        bets = [(t_h, p_h, c_h), ("NUL", p_n, c_n), (t_a, p_a, c_a)]
        summary = []
        for lab, p, c in bets:
            ev = p * c
            if ev > 1.05:
                k = ((c-1)*p - (1-p)) / (c-1)
                mise = max(0, bankroll * k * risk_map[risk_mode])
                summary.append({"PARI": lab, "PROBABILITE": f"{p*100:.1f}%", "VALUE": f"{ev:.2f}", "MISE": f"{mise:.2f}€"})
        
        if summary: st.table(pd.DataFrame(summary))
        else: st.write("AUCUNE VALUE DETECTEE")
        st.markdown("</div>", unsafe_allow_html=True)

        # Scores Exacts
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.subheader("SCORES EXACTS")
        flat = matrix.flatten()
        top3 = np.argsort(flat)[-3:][::-1]
        sc_cols = st.columns(3)
        for i, idx in enumerate(top3):
            h, a = divmod(idx, 6)
            sc_cols[i].metric(f"{h} - {a}", f"{matrix[h,a]*100:.1f}%")
        st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<p style='text-align:center; opacity:0.4; letter-spacing:3px;'>ITROZ MASTER EDITION</p>", unsafe_allow_html=True)
