import streamlit as st
import requests
import numpy as np
import pandas as pd
from scipy.stats import poisson

# --- CONFIGURATION PAGE ---
st.set_page_config(page_title="iTrOz Predictor Master", layout="wide")

# --- STYLE CSS (TITRES VISIBLES ET COHERENCE) ---
st.markdown("""
    <style>
    .stApp {
        background-image: url("https://media.giphy.com/media/VZrfUvQjXaGEQy1RSn/giphy.gif");
        background-size: cover;
        background-attachment: fixed;
    }
    .stApp > div:first-child { background-color: rgba(0, 0, 0, 0.92); }
    
    /* Titres des catégories bien dorés et visibles */
    .cat-title {
        color: #FFD700 !important;
        font-family: 'Monospace', sans-serif;
        font-size: 1.4rem;
        font-weight: bold;
        letter-spacing: 3px;
        border-left: 4px solid #FFD700;
        padding-left: 15px;
        margin-top: 30px;
        margin-bottom: 15px;
        text-transform: uppercase;
    }

    h1, p, span, label, .stMetric div { 
        color: #FFD700 !important; 
        font-family: 'Monospace', sans-serif !important; 
    }

    .glass-card {
        background: rgba(0, 0, 0, 0.6) !important;
        border: 1px solid rgba(255, 215, 0, 0.2);
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
    }

    div[data-baseweb="select"], div[data-baseweb="input"], .stNumberInput input {
        background-color: rgba(0, 0, 0, 0.9) !important;
        border: 1px solid rgba(255, 215, 0, 0.4) !important;
        color: #FFD700 !important;
    }

    div.stButton > button {
        background: #FFD700 !important;
        color: #000 !important;
        font-weight: bold;
        width: 100%;
        height: 3.5em;
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

if 'results' not in st.session_state:
    st.session_state.results = None

st.markdown("<h1 style='text-align:center; letter-spacing:15px;'>ITROZ PREDICTOR</h1>", unsafe_allow_html=True)

# 1. PARAMETRES
st.markdown("<div class='cat-title'>1. CONFIGURATION</div>", unsafe_allow_html=True)
st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
c_bank, c_risk = st.columns([1, 2])
bankroll = c_bank.number_input("CAPITAL (€)", value=100.0)
risk_mode = c_risk.radio("MODE DE MISE", ["🛡️ SAFE", "⚖️ MID", "🔥 JOUEUR"], horizontal=True)
risk_map = {"🛡️ SAFE": 0.1, "⚖️ MID": 0.25, "🔥 JOUEUR": 0.5}
st.markdown("</div>", unsafe_allow_html=True)

# 2. SELECTION
st.markdown("<div class='cat-title'>2. MATCH</div>", unsafe_allow_html=True)
st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
leagues = {"Premier League": 39, "Ligue 1": 61, "La Liga": 140, "Bundesliga": 78, "Serie A": 135}
col_l, col_h, col_a = st.columns(3)
l_id = leagues[col_l.selectbox("CHAMPIONNAT", list(leagues.keys()))]
teams_data = call_api("teams", {"league": l_id, "season": SEASON})
team_map = {t['team']['name']: t['team']['id'] for t in teams_data}
t_h = col_h.selectbox("DOMICILE", sorted(team_map.keys()))
t_a = col_a.selectbox("EXTERIEUR", sorted(team_map.keys()), index=1 if len(team_map)>1 else 0)

if st.button("LANCER L'ANALYSE GLOBALE"):
    with st.spinner("CALCUL DES FLUX..."):
        m_h, s_h = get_stats(team_map[t_h], l_id)
        m_a, s_a = get_stats(team_map[t_a], l_id)
        lh, la = m_h * (m_a/1.3), m_a * (m_h/1.3)
        matrix = np.outer(poisson.pmf(np.arange(6), lh), poisson.pmf(np.arange(6), la))
        matrix /= matrix.sum()
        st.session_state.results = {
            "p_h": np.sum(np.tril(matrix, -1)), "p_n": np.sum(np.diag(matrix)), "p_a": np.sum(np.triu(matrix, 1)),
            "p_1n": np.sum(np.tril(matrix, -1)) + np.sum(np.diag(matrix)),
            "p_n2": np.sum(np.triu(matrix, 1)) + np.sum(np.diag(matrix)),
            "p_12": np.sum(np.tril(matrix, -1)) + np.sum(np.triu(matrix, 1)),
            "p_btts": np.sum(matrix[1:, 1:]), "p_o25": sum(matrix[i, j] for i in range(6) for j in range(6) if i+j > 2.5),
            "matrix": matrix, "t_h": t_h, "t_a": t_a
        }
st.markdown("</div>", unsafe_allow_html=True)

if st.session_state.results:
    res = st.session_state.results
    
    # 3. PROBABILITES
    st.markdown("<div class='cat-title'>3. PROBABILITES CALCULEES</div>", unsafe_allow_html=True)
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    r1, r2, r3, r4, r5 = st.columns(5)
    r1.metric(res['t_h'], f"{res['p_h']*100:.1f}%")
    r2.metric("NUL", f"{res['p_n']*100:.1f}%")
    r3.metric(res['t_a'], f"{res['p_a']*100:.1f}%")
    r4.metric("BTTS", f"{res['p_btts']*100:.1f}%")
    r5.metric("OVER 2.5", f"{res['p_o25']*100:.1f}%")
    st.markdown("</div>", unsafe_allow_html=True)

    # 4. VALEURS ET COTES (BET)
    st.markdown("<div class='cat-title'>4. DETECTION DE VALUE (BET)</div>", unsafe_allow_html=True)
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.write("Saisissez les cotes pour voir les opportunités :")
    bc1, bc2, bc3, bc4, bc5, bc6 = st.columns(6)
    c_h = bc1.number_input(f"Cote {res['t_h']}", value=1.0)
    c_n = bc2.number_input("Cote NUL", value=1.0)
    c_a = bc3.number_input(f"Cote {res['t_a']}", value=1.0)
    c_1n = bc4.number_input("Cote 1N", value=1.0)
    c_n2 = bc5.number_input("Cote N2", value=1.0)
    c_btts = bc6.number_input("Cote BTTS", value=1.0)
    
    all_bets = [
        (f"Victoire {res['t_h']}", res['p_h'], c_h), ("Nul", res['p_n'], c_n), (f"Victoire {res['t_a']}", res['p_a'], c_a),
        ("Double 1N", res['p_1n'], c_1n), ("Double N2", res['p_n2'], c_n2), ("BTTS Oui", res['p_btts'], c_btts)
    ]
    summary = []
    for lab, p, c in all_bets:
        if c > 1.0 and (p * c) > 1.05:
            k = ((c-1)*p - (1-p)) / (c-1)
            summary.append({"PARI": lab, "VALUE": f"{p*c:.2f}", "MISE": f"{max(0, bankroll * k * risk_map[risk_mode]):.2f}€"})
    
    if summary: st.table(pd.DataFrame(summary))
    else: st.info("Saisissez les cotes réelles pour détecter la Value.")
    st.markdown("</div>", unsafe_allow_html=True)

    # 5. AUDIT
    st.markdown("<div class='cat-title'>5. AUDIT DU TICKET PERSONNEL</div>", unsafe_allow_html=True)
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    col_a1, col_a2, col_a3 = st.columns([2,1,1])
    choix = col_a1.selectbox("PRONO A TESTER", [f"Victoire : {res['t_h']}", "Match Nul", f"Victoire : {res['t_a']}", "Double Chance 1N", "Double Chance N2", "BTTS (Oui)", "Over 2.5"])
    cote_audit = col_a2.number_input("VOTRE COTE", value=1.80)
    
    audit_map = {
        f"Victoire : {res['t_h']}": res['p_h'], "Match Nul": res['p_n'], f"Victoire : {res['t_a']}": res['p_a'],
        "Double Chance 1N": res['p_1n'], "Double Chance N2": res['p_n2'], "BTTS (Oui)": res['p_btts'], "Over 2.5": res['p_o25']
    }
    ev = audit_map[choix] * cote_audit
    col_a3.markdown(f"<br><span style='color:{'#00FF41' if ev >= 1.05 else '#FF4B4B'}; font-size:1.5rem; font-weight:bold;'>{'VALIDE' if ev >= 1.05 else 'RISQUE'} ({ev:.2f})</span>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # 6. SCORES
    st.markdown("<div class='cat-title'>6. SCORES LES PLUS PROBABLES</div>", unsafe_allow_html=True)
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    flat = res['matrix'].flatten()
    top3 = np.argsort(flat)[-3:][::-1]
    sc1, sc2, sc3 = st.columns(3)
    cols = [sc1, sc2, sc3]
    for i, idx in enumerate(top3):
        h, a = divmod(idx, 6)
        cols[i].metric(f"SCORE {h}-{a}", f"{res['matrix'][h,a]*100:.1f}%")
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<p style='text-align:center; opacity:0.3; letter-spacing:5px;'>ITROZ SYSTEM v11.0</p>", unsafe_allow_html=True)
