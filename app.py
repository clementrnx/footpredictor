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
        transition: 0.3s;
        width: 100%;
    }
    
    div.stButton > button:hover { 
        background: #FFD700 !important; 
        color: #000 !important;
        box-shadow: 0 0 20px rgba(255, 215, 0, 0.4);
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
st.markdown("<h1 style='text-align:center; letter-spacing:15px; margin-bottom:40px;'>ITROZ PREDICTOR</h1>", unsafe_allow_html=True)

# 1. PARAMETRES
st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
st.markdown("### 1. SESSION")
c_bank, c_risk = st.columns([1, 2])
bankroll = c_bank.number_input("CAPITAL (€)", value=100.0)
risk_mode = c_risk.radio("RISQUE", ["🛡️ SAFE", "⚖️ MID", "🔥 JOUEUR"], horizontal=True)
risk_map = {"🛡️ SAFE": 0.1, "⚖️ MID": 0.25, "🔥 JOUEUR": 0.5}
st.markdown("</div>", unsafe_allow_html=True)

# 2. MATCH
st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
st.markdown("### 2. SELECTION")
leagues = {"Premier League": 39, "Ligue 1": 61, "La Liga": 140, "Bundesliga": 78, "Serie A": 135}
col_l, col_h, col_a = st.columns(3)
l_name = col_l.selectbox("LIGUE", list(leagues.keys()))
teams_data = call_api("teams", {"league": leagues[l_name], "season": SEASON})
team_map = {t['team']['name']: t['team']['id'] for t in teams_data}
t_h = col_h.selectbox("DOMICILE", sorted(team_map.keys()))
t_a = col_a.selectbox("EXTERIEUR", sorted(team_map.keys()), index=1)
st.markdown("</div>", unsafe_allow_html=True)

if st.button("LANCER L'ANALYSE"):
    with st.spinner("CALCUL DES FLUX..."):
        m_h, s_h = get_stats(team_map[t_h], leagues[l_name])
        m_a, s_a = get_stats(team_map[t_a], leagues[l_name])
        
        lh, la = m_h * (m_a/1.3), m_a * (m_h/1.3)
        matrix = np.outer(poisson.pmf(np.arange(6), lh), poisson.pmf(np.arange(6), la))
        matrix /= matrix.sum()
        
        p_h, p_n, p_a = np.sum(np.tril(matrix, -1)), np.sum(np.diag(matrix)), np.sum(np.triu(matrix, 1))
        p_btts = np.sum(matrix[1:, 1:])
        p_o25 = sum(matrix[i, j] for i in range(6) for j in range(6) if i+j > 2.5)

        # 3. PROBAS
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("### 3. PROBABILITES")
        r1, r2, r3 = st.columns(3)
        r1.metric(t_h.upper(), f"{p_h*100:.1f}%")
        r2.metric("NUL", f"{p_n*100:.1f}%")
        r3.metric(t_a.upper(), f"{p_a*100:.1f}%")
        st.markdown("</div>", unsafe_allow_html=True)

        # 4. BETS (VALUE DETECTION AUTO)
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("### 4. OPPORTUNITES DETECTEES (VALUE)")
        vc1, vc2, vc3 = st.columns(3)
        bk_h = vc1.number_input(f"COTE {t_h}", value=2.0)
        bk_n = vc2.number_input("COTE NUL", value=3.2)
        bk_a = vc3.number_input(f"COTE {t_a}", value=3.5)
        
        auto_bets = [(t_h, p_h, bk_h), ("NUL", p_n, bk_n), (t_a, p_a, bk_a)]
        summary = []
        for lab, p, c in auto_bets:
            if p * c > 1.05:
                k = ((c-1)*p - (1-p)) / (c-1)
                mise = max(0, bankroll * k * risk_map[risk_mode])
                summary.append({"CHOIX": lab, "PROBA": f"{p*100:.1f}%", "VALUE": f"{p*c:.2f}", "MISE": f"{mise:.2f}€"})
        
        if summary: st.table(pd.DataFrame(summary))
        else: st.info("AUCUNE VALUE SUR LE 1N2")
        st.markdown("</div>", unsafe_allow_html=True)

        # 5. AUDIT DU TICKET
        st.markdown("<div class='glass-card' style='border: 1px solid #FFD700;'>", unsafe_allow_html=True)
        st.markdown("### 5. AUDIT DU TICKET PERSONNEL")
        ac1, ac2, ac3 = st.columns([2, 1, 1])
        type_bet = ac1.selectbox("PRONO", ["VICTOIRE DOMICILE", "MATCH NUL", "VICTOIRE EXTERIEUR", "BTTS (OUI)", "OVER 2.5"])
        cote_jouee = ac2.number_input("COTE", value=1.80)
        
        audit_map = {"VICTOIRE DOMICILE": p_h, "MATCH NUL": p_n, "VICTOIRE EXTERIEUR": p_a, "BTTS (OUI)": p_btts, "OVER 2.5": p_o25}
        target_p = audit_map[type_bet]
        ev_audit = target_p * cote_jouee
        
        if ev_audit >= 1.05:
            ac3.markdown(f"<br><span style='color:#00FF41;'>VALIDE ({ev_audit:.2f})</span>", unsafe_allow_html=True)
        else:
            ac3.markdown(f"<br><span style='color:#FF4B4B;'>RISQUE ({ev_audit:.2f})</span>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # 6. SCORES
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("### 6. SCORES EXACTS")
        flat = matrix.flatten()
        top3 = np.argsort(flat)[-3:][::-1]
        sc_cols = st.columns(3)
        for i, idx in enumerate(top3):
            h, a = divmod(idx, 6)
            sc_cols[i].metric(f"{h} - {a}", f"{matrix[h,a]*100:.1f}%")
        st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<p style='text-align:center; opacity:0.3; letter-spacing:5px;'>ITROZ SYSTEM v8.0</p>", unsafe_allow_html=True)
