import streamlit as st
import requests
import numpy as np
import pandas as pd
from scipy.stats import poisson

# --- CONFIGURATION PAGE ---
st.set_page_config(page_title="iTrOz Predictor Master", layout="wide")

# --- STYLE CSS (PUR ET FIN) ---
st.markdown("""
    <style>
    .stApp {
        background-image: url("https://media.giphy.com/media/VZrfUvQjXaGEQy1RSn/giphy.gif");
        background-size: cover;
        background-attachment: fixed;
    }
    .stApp > div:first-child { background-color: rgba(0, 0, 0, 0.85); }
    
    h1, h2, h3, p, span, label, .stMetric div { 
        color: #FFD700 !important; 
        font-family: 'Monospace', sans-serif !important; 
        text-transform: uppercase;
    }

    /* Conteneurs ultra-légers sans barres massives */
    .glass-card {
        background: rgba(0, 0, 0, 0.4) !important;
        border-bottom: 1px solid rgba(255, 215, 0, 0.3);
        padding: 20px 0px;
        margin-bottom: 10px;
    }

    div[data-baseweb="select"], div[data-baseweb="input"], .stNumberInput input, div[data-baseweb="radio"] {
        background-color: rgba(0, 0, 0, 0.8) !important;
        border: 1px solid rgba(255, 215, 0, 0.2) !important;
        color: #FFD700 !important;
    }

    div.stButton > button {
        background: transparent !important;
        border: 1px solid #FFD700 !important;
        color: #FFD700 !important;
        height: 3.5em;
        font-weight: bold;
        width: 100%;
        margin-top: 20px;
    }
    
    div.stButton > button:hover { 
        background: #FFD700 !important; 
        color: #000 !important;
    }

    hr { border-top: 1px solid rgba(255, 215, 0, 0.1); }
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

if 'results' not in st.session_state:
    st.session_state.results = None

st.markdown("<h1 style='text-align:center; letter-spacing:15px;'>ITROZ PREDICTOR</h1>", unsafe_allow_html=True)

# 1. SESSION
st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
c_bank, c_risk = st.columns([1, 2])
bankroll = c_bank.number_input("CAPITAL (€)", value=100.0)
risk_mode = c_risk.radio("STRATEGIE", ["🛡️ SAFE", "⚖️ MID", "🔥 JOUEUR"], horizontal=True)
risk_map = {"🛡️ SAFE": 0.1, "⚖️ MID": 0.25, "🔥 JOUEUR": 0.5}
st.markdown("</div>", unsafe_allow_html=True)

# 2. SELECTION
st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
leagues = {"Premier League": 39, "Ligue 1": 61, "La Liga": 140, "Bundesliga": 78, "Serie A": 135}
col_l, col_h, col_a = st.columns(3)
l_id = leagues[col_l.selectbox("LIGUE", list(leagues.keys()))]
teams_data = call_api("teams", {"league": l_id, "season": SEASON})
team_map = {t['team']['name']: t['team']['id'] for t in teams_data}
t_h = col_h.selectbox("DOMICILE", sorted(team_map.keys()))
t_a = col_a.selectbox("EXTERIEUR", sorted(team_map.keys()), index=min(1, len(team_map)-1))

if st.button("LANCER L'ANALYSE GLOBALE"):
    with st.spinner("TRAITEMENT..."):
        m_h, s_h = get_stats(team_map[t_h], l_id)
        m_a, s_a = get_stats(team_map[t_a], l_id)
        lh, la = m_h * (m_a/1.3), m_a * (m_h/1.3)
        matrix = np.outer(poisson.pmf(np.arange(6), lh), poisson.pmf(np.arange(6), la))
        matrix /= matrix.sum()
        st.session_state.results = {
            "p_h": np.sum(np.tril(matrix, -1)), "p_n": np.sum(np.diag(matrix)), "p_a": np.sum(np.triu(matrix, 1)),
            "p_btts": np.sum(matrix[1:, 1:]), "p_o25": sum(matrix[i, j] for i in range(6) for j in range(6) if i+j > 2.5),
            "matrix": matrix, "t_h": t_h, "t_a": t_a
        }
st.markdown("</div>", unsafe_allow_html=True)

if st.session_state.results:
    res = st.session_state.results
    
    # 3. PROBAS 1N2
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    r1, r2, r3 = st.columns(3)
    r1.metric(res['t_h'], f"{res['p_h']*100:.1f}%")
    r2.metric("NUL", f"{res['p_n']*100:.1f}%")
    r3.metric(res['t_a'], f"{res['p_a']*100:.1f}%")
    st.markdown("</div>", unsafe_allow_html=True)

    # 4. VALUE AUTO
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    v1, v2, v3 = st.columns(3)
    c_h = v1.number_input(f"COTE {res['t_h']}", value=2.0)
    c_n = v2.number_input("COTE NUL", value=3.2)
    c_a = v3.number_input(f"COTE {res['t_a']}", value=3.5)
    
    summary = []
    for lab, p, c in [(res['t_h'], res['p_h'], c_h), ("NUL", res['p_n'], c_n), (res['t_a'], res['p_a'], c_a)]:
        if p * c > 1.05:
            k = ((c-1)*p - (1-p)) / (c-1)
            mise = max(0, bankroll * k * risk_map[risk_mode])
            summary.append({"PARI": lab, "VALUE": f"{p*c:.2f}", "MISE": f"{mise:.2f}€"})
    if summary: st.table(pd.DataFrame(summary))
    st.markdown("</div>", unsafe_allow_html=True)

    # 5. AUDIT
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    col_a1, col_a2, col_a3 = st.columns([2,1,1])
    choix = col_a1.selectbox("AUDIT TICKET", [f"Victoire : {res['t_h']}", "Match Nul", f"Victoire : {res['t_a']}", "BTTS (Oui)", "Over 2.5"])
    cote_audit = col_a2.number_input("COTE JOUEE", value=1.80)
    audit_map = {f"Victoire : {res['t_h']}": res['p_h'], "Match Nul": res['p_n'], f"Victoire : {res['t_a']}": res['p_a'], "BTTS (Oui)": res['p_btts'], "Over 2.5": res['p_o25']}
    ev = audit_map[choix] * cote_audit
    col_a3.markdown(f"<br><span style='color:{'#00FF41' if ev >= 1.05 else '#FF4B4B'}; font-weight:bold;'>{'VALIDE' if ev >= 1.05 else 'RISQUE'} ({ev:.2f})</span>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # 6. SCORES & SPECIAUX
    c_sc, c_sp = st.columns(2)
    with c_sc:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        top3 = np.argsort(res['matrix'].flatten())[-3:][::-1]
        for idx in top3:
            h, a = divmod(idx, 6)
            st.write(f"SCORE {h}-{a} : **{res['matrix'][h,a]*100:.1f}%**")
        st.markdown("</div>", unsafe_allow_html=True)
    with c_sp:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.write(f"BTTS : **{res['p_btts']*100:.1f}%**")
        st.write(f"OVER 2.5 : **{res['p_o25']*100:.1f}%**")
        st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<p style='text-align:center; opacity:0.3; letter-spacing:5px;'>ITROZ MASTER v10.0</p>", unsafe_allow_html=True)
