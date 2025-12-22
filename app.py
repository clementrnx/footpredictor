import streamlit as st
import requests
import numpy as np
from scipy.stats import poisson

st.set_page_config(page_title="iTrOz Predictor", layout="wide")

st.markdown("""
    <style>
    .stApp {
        background-image: url("https://media.giphy.com/media/VZrfUvQjXaGEQy1RSn/giphy.gif");
        background-size: cover;
        background-attachment: fixed;
    }
    .stApp > div:first-child { background-color: rgba(0, 0, 0, 0.90); }
    
    h1, h2, h3, p, span, label { color: #FFD700 !important; font-family: 'Monospace', sans-serif; letter-spacing: 2px; }

    /* BOUTONS LATÉRAUX LONGS ET TRANSLUCIDES */
    div.stButton > button {
        background: rgba(255, 215, 0, 0.02) !important;
        backdrop-filter: blur(25px) !important;
        border: 1px solid rgba(255, 215, 0, 0.1) !important;
        color: #FFD700 !important;
        border-radius: 0px !important;
        height: 60px !important;
        width: 100% !important;
        font-weight: 200 !important;
        text-transform: uppercase !important;
        letter-spacing: 15px !important;
        transition: 0.8s;
        border-left: none !important;
        border-right: none !important;
    }
    
    div.stButton > button:hover { 
        background: rgba(255, 215, 0, 0.1) !important;
        border: 1px solid rgba(255, 215, 0, 0.4) !important;
        letter-spacing: 20px !important;
        box-shadow: 0 0 50px rgba(255, 215, 0, 0.1);
    }

    /* TYPOGRAPHIE VERDICT */
    .verdict-text {
        font-size: 32px;
        font-weight: 900;
        text-align: center;
        padding: 40px;
        letter-spacing: 10px;
        text-transform: uppercase;
    }

    div[data-baseweb="select"] > div, div[data-baseweb="input"] > div {
        background: transparent !important; 
        border-bottom: 1px solid rgba(255, 215, 0, 0.2) !important;
    }

    .bet-card {
        background: rgba(255, 255, 255, 0.01);
        padding: 40px;
        margin-bottom: 50px;
    }
    </style>
""", unsafe_allow_html=True)

API_KEY = st.secrets["MY_API_KEY"]
BASE_URL = "https://v3.football.api-sports.io/"
HEADERS = {'x-apisports-key': API_KEY}
SEASON = 2025

@st.cache_data(ttl=3600)
def get_api(endpoint, params):
    try:
        r = requests.get(f"{BASE_URL}{endpoint}", headers=HEADERS, params=params, timeout=12)
        return r.json().get('response', [])
    except: return []

if 'simulation_done' not in st.session_state:
    st.session_state.simulation_done = False
    st.session_state.data = {}

st.title("ITROZ PREDICTOR")

leagues = {"Champions League": 2, "Premier League": 39, "La Liga": 140, "Serie A": 135, "Bundesliga": 78, "Ligue 1": 61}
l_name = st.selectbox("LIGUE", list(leagues.keys()))
l_id = leagues[l_name]

teams_res = get_api("teams", {"league": l_id, "season": SEASON})
teams = {t['team']['name']: t['team']['id'] for t in teams_res}

if teams:
    c1, c2 = st.columns(2)
    t_h = c1.selectbox("HOME", sorted(teams.keys()))
    t_a = c2.selectbox("AWAY", sorted(teams.keys()), index=1)

    if st.button("CALCULATE"):
        id_h, id_a = teams[t_h], teams[t_a]
        s_h = get_api("teams/statistics", {"league": l_id, "season": SEASON, "team": id_h})
        s_a = get_api("teams/statistics", {"league": l_id, "season": SEASON, "team": id_a})
        
        if s_h and s_a:
            att_h = float(s_h.get('goals',{}).get('for',{}).get('average',{}).get('total', 1.3))
            def_a = float(s_a.get('goals',{}).get('against',{}).get('average',{}).get('total', 1.3))
            att_a = float(s_a.get('goals',{}).get('for',{}).get('average',{}).get('total', 1.3))
            def_h = float(s_h.get('goals',{}).get('against',{}).get('average',{}).get('total', 1.3))
            lh, la = (att_h * def_a / 1.3) * 1.12, (att_a * def_h / 1.3)
            matrix = np.zeros((7, 7))
            for x in range(7):
                for y in range(7): matrix[x,y] = poisson.pmf(x, lh) * poisson.pmf(y, la)
            matrix /= matrix.sum()
            st.session_state.data = {'p_h': np.sum(np.tril(matrix, -1)), 'p_n': np.sum(np.diag(matrix)), 'p_a': np.sum(np.triu(matrix, 1)), 'matrix': matrix, 't_h': t_h, 't_a': t_a}
            st.session_state.simulation_done = True

if st.session_state.simulation_done:
    d = st.session_state.data
    st.write("---")
    
    # --- SECTION 1 : MODE BET (IA RECOM) ---
    st.subheader("🤖 MODE BET")
    st.markdown("<div class='bet-card'>", unsafe_allow_html=True)
    
    col_cap, col_1, col_N, col_2 = st.columns(4)
    bankroll = col_cap.number_input("BANKROLL (€)", value=100.0)
    c_h = col_1.number_input(f"COTE {d['t_h']}", value=2.0)
    c_n = col_N.number_input("COTE NUL", value=3.0)
    c_a = col_2.number_input(f"COTE {d['t_a']}", value=3.0)

    col_dc1, col_dc2, col_dc3 = st.columns(3)
    c_hn = col_dc1.number_input(f"{d['t_h']}/NUL", value=1.30)
    c_na = col_dc2.number_input(f"NUL/{d['t_a']}", value=1.30)
    c_ha = col_dc3.number_input(f"{d['t_h']}/{d['t_a']}", value=1.30)

    options = [
        {"n": d['t_h'], "p": d['p_h'], "c": c_h},
        {"n": "NUL", "p": d['p_n'], "c": c_n},
        {"n": d['t_a'], "p": d['p_a'], "c": c_a},
        {"n": f"{d['t_h']} OU NUL", "p": d['p_h'] + d['p_n'], "c": c_hn},
        {"n": f"NUL OU {d['t_a']}", "p": d['p_n'] + d['p_a'], "c": c_na},
        {"n": f"{d['t_h']} OU {d['t_a']}", "p": d['p_h'] + d['p_a'], "c": c_ha}
    ]

    best = max(options, key=lambda o: o['p'] * o['c'])
    if best['p'] * best['c'] > 1.02:
        b = best['c'] - 1
        k = ((b * best['p']) - (1 - best['p'])) / b if b > 0 else 0
        mise = max(bankroll * 0.30, bankroll * k * 0.5)
        mise = min(mise, bankroll * 0.25)
        st.markdown(f"<div class='verdict-text'>IA RECOMMANDE : {best['n']} | MISE : {mise:.2f}€</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='verdict-text'>AUCUN VALUE DÉTECTÉ</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # --- SECTION 2 : AUDIT ---
    st.subheader("🔍 AUDIT DU TICKET")
    a1, a2 = st.columns(2)
    choix = a1.selectbox("VOTRE PARI", [d['t_h'], "Nul", d['t_a'], f"{d['t_h']} ou Nul", f"Nul ou {d['t_a']}", f"{d['t_h']} ou {d['t_a']}"])
    cote_audit = a2.number_input("VOTRE COTE", value=1.50, step=0.01)

    prob_ia = d['p_h'] if choix == d['t_h'] else (d['p_n'] if choix == "Nul" else d['p_a'])
    if "ou Nul" in choix and d['t_h'] in choix: prob_ia = d['p_h'] + d['p_n']
    elif "Nul ou" in choix: prob_ia = d['p_n'] + d['p_a']
    elif "ou" in choix: prob_ia = d['p_h'] + d['p_a']
    
    val = prob_ia * cote_audit
    v_status = "SAFE" if val >= 1.10 else ("MID" if val >= 0.98 else "DANGEREUX")
    st.markdown(f"<div class='verdict-text'>AUDIT : {v_status} (VALUE: {val:.2f})</div>", unsafe_allow_html=True)

    # --- SECTION 3 : SCORES ---
    st.subheader("PROBABLE SCORES")
    idx = np.unravel_index(np.argsort(d['matrix'].ravel())[-3:][::-1], d['matrix'].shape)
    s1, s2, s3 = st.columns(3)
    for i in range(3):
        with [s1, s2, s3][i]: st.write(f"**{idx[0][i]} - {idx[1][i]}** ({d['matrix'][idx[0][i], idx[1][i]]*100:.1f}%)")
