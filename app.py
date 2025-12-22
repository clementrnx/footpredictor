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
    .stApp > div:first-child { background-color: rgba(0, 0, 0, 0.85); }
    
    h1, h2, h3, p, span, label { color: #FFD700 !important; font-family: 'Monospace', sans-serif; }

    /* BOUTONS ULTRAS TRANSLUCIDES ET LONGS */
    div.stButton > button {
        background: rgba(255, 215, 0, 0.05) !important;
        backdrop-filter: blur(20px) !important;
        -webkit-backdrop-filter: blur(20px) !important;
        border: 1px solid rgba(255, 215, 0, 0.2) !important;
        color: #FFD700 !important;
        border-radius: 5px !important;
        height: 50px !important;
        width: 100% !important;
        font-weight: 300 !important;
        text-transform: uppercase !important;
        letter-spacing: 8px !important;
        transition: 0.5s;
    }
    
    div.stButton > button:hover { 
        background: rgba(255, 215, 0, 0.15) !important;
        border: 1px solid rgba(255, 215, 0, 0.5) !important;
        letter-spacing: 12px !important;
    }

    /* INPUTS SANS BORDURES */
    div[data-baseweb="select"], div[data-baseweb="input"], input { background: transparent !important; border: none !important; }
    div[data-baseweb="select"] > div, div[data-baseweb="input"] > div {
        background: transparent !important; border-bottom: 1px solid rgba(255, 215, 0, 0.3) !important; border-radius: 0px !important;
    }

    .bet-card {
        background: rgba(255, 255, 255, 0.02);
        backdrop-filter: blur(15px);
        border-left: 2px solid #FFD700;
        padding: 30px;
        margin-top: 25px;
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

st.title("iTrOz Predictor")

leagues = {"Champions League": 2, "Premier League": 39, "La Liga": 140, "Serie A": 135, "Bundesliga": 78, "Ligue 1": 61}
l_name = st.selectbox("COMPÉTITION", list(leagues.keys()))
l_id = leagues[l_name]

teams_res = get_api("teams", {"league": l_id, "season": SEASON})
teams = {t['team']['name']: t['team']['id'] for t in teams_res}

if teams:
    c1, c2 = st.columns(2)
    t_h = c1.selectbox("DOMICILE", sorted(teams.keys()))
    t_a = c2.selectbox("EXTÉRIEUR", sorted(teams.keys()), index=1)

    if st.button("Lancer l'Analyse"):
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
    
    # METRICS EN LIGNE
    r1, r2, r3 = st.columns(3)
    r1.metric(d['t_h'], f"{d['p_h']*100:.1f}%")
    r2.metric("NUL", f"{d['p_n']*100:.1f}%")
    r3.metric(d['t_a'], f"{d['p_a']*100:.1f}%")

    st.write("### AUDIT DU TICKET")
    a1, a2, a3 = st.columns(3)
    choix = a1.selectbox("PARI", [d['t_h'], "Nul", d['t_a'], f"{d['t_h']} ou Nul", f"Nul ou {d['t_a']}", f"{d['t_h']} ou {d['t_a']}"])
    cote_audit = a2.number_input("COTE", value=1.50, step=0.01)
    bankroll = a3.number_input("Capital (€)", value=100.0)

    # CALCUL IMMEDIAT SANS BOUTON
    prob_ia = d['p_h'] if choix == d['t_h'] else (d['p_n'] if choix == "Nul" else d['p_a'])
    if "ou Nul" in choix and d['t_h'] in choix: prob_ia = d['p_h'] + d['p_n']
    elif "Nul ou" in choix: prob_ia = d['p_n'] + d['p_a']
    elif "ou" in choix: prob_ia = d['p_h'] + d['p_a']
    
    ev = prob_ia * cote_audit
    if ev >= 1.10: st.success(f"VERDICT : SAFE (EV: {ev:.2f})")
    elif ev >= 0.98: st.warning(f"VERDICT : MID (EV: {ev:.2f})")
    else: st.error(f"VERDICT : ENLÈVE (EV: {ev:.2f})")

    st.markdown("<div class='bet-card'>", unsafe_allow_html=True)
    st.subheader("🤖 MODE BET")
    
    st.write("**COTES RÉELLES**")
    bt1, bt2, bt3 = st.columns(3)
    c_h, c_n, c_a = bt1.number_input(f"Cote {d['t_h']}", value=2.0), bt2.number_input("Cote NUL", value=3.0), bt3.number_input(f"Cote {d['t_a']}", value=3.0)

    bt4, bt5, bt6 = st.columns(3)
    c_hn, c_na, c_ha = bt4.number_input(f"{d['t_h']}/Nul", value=1.30), bt5.number_input(f"Nul/{d['t_a']}", value=1.30), bt6.number_input(f"{d['t_h']}/{d['t_a']}", value=1.30)

    options = [
        {"n": d['t_h'], "p": d['p_h'], "c": c_h},
        {"n": "Nul", "p": d['p_n'], "c": c_n},
        {"n": d['t_a'], "p": d['p_a'], "c": c_a},
        {"n": f"{d['t_h']} ou Nul", "p": d['p_h'] + d['p_n'], "c": c_hn},
        {"n": f"Nul ou {d['t_a']}", "p": d['p_n'] + d['p_a'], "c": c_na},
        {"n": f"{d['t_h']} ou {d['t_a']}", "p": d['p_h'] + d['p_a'], "c": c_ha}
    ]

    best = max(options, key=lambda o: o['p'] * o['c'])
    if best['p'] * best['c'] > 1.03:
        b = best['c'] - 1
        kelly = ((b * best['p']) - (1 - best['p'])) / b if b > 0 else 0
        mise = max(bankroll * 0.30, bankroll * kelly * 0.5)
        mise = min(mise, bankroll * 0.25)
        st.write(f"### 🚀 VERDICT : **{best['n'].upper()}** | Mise : **{mise:.2f}€**")
    else: st.write("### ❌ AUCUN VALUE")
    st.markdown("</div>", unsafe_allow_html=True)

    st.write("### SCORES PROBABLES")
    idx = np.unravel_index(np.argsort(d['matrix'].ravel())[-3:][::-1], d['matrix'].shape)
    s1, s2, s3 = st.columns(3)
    for i in range(3):
        with [s1, s2, s3][i]: st.write(f"**{idx[0][i]} - {idx[1][i]}** ({d['matrix'][idx[0][i], idx[1][i]]*100:.1f}%)")
