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
    .stApp > div:first-child { background-color: rgba(0, 0, 0, 0.92); }
    
    h1, h2, h3, p, span, label { color: #FFD700 !important; font-family: 'Monospace', sans-serif; letter-spacing: 2px; }

    /* BOUTON LONG ET PRESQUE INVISIBLE AU REPOS */
    div.stButton > button {
        background: rgba(255, 215, 0, 0.01) !important;
        backdrop-filter: blur(30px) !important;
        -webkit-backdrop-filter: blur(30px) !important;
        border: 0.1px solid rgba(255, 215, 0, 0.1) !important;
        color: #FFD700 !important;
        border-radius: 0px !important;
        height: 70px !important;
        width: 100% !important;
        text-transform: uppercase !important;
        letter-spacing: 15px !important;
        transition: 0.8s all ease;
        margin-top: 20px;
    }
    
    div.stButton > button:hover { 
        background: rgba(255, 215, 0, 0.08) !important;
        letter-spacing: 20px !important;
        border: 1px solid rgba(255, 215, 0, 0.3) !important;
    }

    /* CHAMPS DE SAISIE ULTRA-TRANSPARENTS */
    div[data-baseweb="select"], div[data-baseweb="input"], .stNumberInput input, .stSelectbox div {
        background-color: rgba(255, 255, 255, 0.01) !important;
        backdrop-filter: blur(20px) !important;
        -webkit-backdrop-filter: blur(20px) !important;
        border: 0.1px solid rgba(255, 215, 0, 0.1) !important;
        border-radius: 5px !important;
        color: #FFD700 !important;
    }

    div[data-baseweb="base-input"] {
        background-color: transparent !important;
        border: none !important;
    }

    .verdict-text {
        font-size: 24px;
        font-weight: 300;
        text-align: center;
        padding: 40px;
        letter-spacing: 12px;
        text-transform: uppercase;
        border-top: 0.5px solid rgba(255, 215, 0, 0.1);
        border-bottom: 0.5px solid rgba(255, 215, 0, 0.1);
        margin: 20px 0;
    }

    .bet-card {
        background: rgba(255, 255, 255, 0.005);
        padding: 35px;
        border-radius: 10px;
        border: 0.1px solid rgba(255, 215, 0, 0.05);
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

    if st.button("Lancer la prédiction"):
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
    
    m1, m2, m3 = st.columns(3)
    m1.metric(d['t_h'], f"{d['p_h']*100:.1f}%")
    m2.metric("NUL", f"{d['p_n']*100:.1f}%")
    m3.metric(d['t_a'], f"{d['p_a']*100:.1f}%")

    st.subheader("🤖 MODE BET")
    st.markdown("<div class='bet-card'>", unsafe_allow_html=True)
    
    b1, b2, b3, b4 = st.columns(4)
    bankroll = b1.number_input("BANKROLL (€)", value=100.0)
    c_h = b2.number_input(f"COTE {d['t_h']}", value=2.0)
    c_n = b3.number_input("COTE NUL", value=3.0)
    c_a = b4.number_input(f"COTE {d['t_a']}", value=3.0)

    d1, d2, d3 = st.columns(3)
    c_hn = d1.number_input(f"COTE {d['t_h']}/NUL", value=1.30)
    c_na = d2.number_input(f"COTE NUL/{d['t_a']}", value=1.30)
    c_ha = d3.number_input(f"COTE {d['t_h']}/{d['t_a']}", value=1.30)

    opts = [
        {"n": d['t_h'], "p": d['p_h'], "c": c_h},
        {"n": "NUL", "p": d['p_n'], "c": c_n},
        {"n": d['t_a'], "p": d['p_a'], "c": c_a},
        {"n": f"{d['t_h']} OU NUL", "p": d['p_h'] + d['p_n'], "c": c_hn},
        {"n": f"NUL OU {d['t_a']}", "p": d['p_n'] + d['p_a'], "c": c_na},
        {"n": f"{d['t_h']} OU {d['t_a']}", "p": d['p_h'] + d['p_a'], "c": c_ha}
    ]

    best = max(opts, key=lambda x: x['p'] * x['c'])
    if best['p'] * best['c'] > 1.02:
        b_v = best['c'] - 1
        k_v = ((b_v * best['p']) - (1 - best['p'])) / b_v if b_v > 0 else 0
        m_f = max(bankroll * 0.30, bankroll * k_v * 0.5)
        m_f = min(m_f, bankroll * 0.25)
        st.markdown(f"<div class='verdict-text'>IA : {best['n']} | MISE : {m_f:.2f}€</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='verdict-text'>AUCUN VALUE</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.subheader("🔍 AUDIT")
    a1, a2 = st.columns(2)
    a_ch = a1.selectbox("PARI", [d['t_h'], "Nul", d['t_a'], f"{d['t_h']} ou Nul", f"Nul ou {d['t_a']}", f"{d['t_h']} ou {d['t_a']}"])
    a_co = a2.number_input("COTE", value=1.50)

    p_a = d['p_h'] if a_ch == d['t_h'] else (d['p_n'] if a_ch == "Nul" else d['p_a'])
    if "ou Nul" in a_ch and d['t_h'] in a_ch: p_a = d['p_h'] + d['p_n']
    elif "Nul ou" in a_ch: p_a = d['p_n'] + d['p_a']
    elif "ou" in a_ch: p_a = d['p_h'] + d['p_a']
    
    ev_a = p_a * a_co
    st.markdown(f"<div class='verdict-text'>AUDIT : {'SAFE' if ev_a >= 1.10 else 'MID' if ev_a >= 0.98 else 'DANGEREUX'} (EV: {ev_a:.2f})</div>", unsafe_allow_html=True)

    st.subheader("SCORES")
    idx = np.unravel_index(np.argsort(d['matrix'].ravel())[-3:][::-1], d['matrix'].shape)
    s1, s2, s3 = st.columns(3)
    for i in range(3):
        with [s1, s2, s3][i]: st.write(f"**{idx[0][i]} - {idx[1][i]}** ({d['matrix'][idx[0][i], idx[1][i]]*100:.1f}%)")
