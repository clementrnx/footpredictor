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

    div.stButton > button {
        background: rgba(255, 215, 0, 0.1) !important;
        backdrop-filter: blur(15px) !important;
        -webkit-backdrop-filter: blur(15px) !important;
        border: 1px solid rgba(255, 215, 0, 0.4) !important;
        color: #FFD700 !important;
        border-radius: 10px !important;
        height: 60px !important;
        width: 100% !important;
        font-weight: 900 !important;
        text-transform: uppercase !important;
        letter-spacing: 2px !important;
    }
    
    div.stButton > button:hover { border: 1px solid #FFD700 !important; box-shadow: 0 0 20px rgba(255, 215, 0, 0.2); }

    div[data-baseweb="select"], div[data-baseweb="input"], input { background: transparent !important; border: none !important; }
    div[data-baseweb="select"] > div, div[data-baseweb="input"] > div {
        background: transparent !important; border-bottom: 2px solid #FFD700 !important; border-radius: 0px !important;
    }

    div[data-testid="stMetric"] { background: transparent !important; border: none !important; }

    .autobet-card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(20px);
        border: 2px solid #FFD700;
        padding: 30px;
        border-radius: 20px;
        margin-top: 30px;
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
    st.divider()
    
    r1, r2, r3 = st.columns(3)
    r1.metric(d['t_h'], f"{d['p_h']*100:.1f}%")
    r2.metric("NUL", f"{d['p_n']*100:.1f}%")
    r3.metric(d['t_a'], f"{d['p_a']*100:.1f}%")

    st.subheader("AUDIT DU TICKET")
    a_col1, a_col2, a_col3 = st.columns(3)
    choix = a_col1.selectbox("PARI", [d['t_h'], "Nul", d['t_a'], f"{d['t_h']} ou Nul", f"Nul ou {d['t_a']}", f"{d['t_h']} ou {d['t_a']}"])
    cote_audit = a_col2.number_input("COTE", value=1.50, step=0.01)
    mise_audit = a_col3.number_input("MISE (€)", value=10)

    if st.button("AJUSTER L'AUDIT"):
        prob_ia = d['p_h'] if choix == d['t_h'] else (d['p_n'] if choix == "Nul" else d['p_a'])
        if "ou Nul" in choix and d['t_h'] in choix: prob_ia = d['p_h'] + d['p_n']
        elif "Nul ou" in choix: prob_ia = d['p_n'] + d['p_a']
        elif "ou" in choix: prob_ia = d['p_h'] + d['p_a']
        
        ev = prob_ia * cote_audit
        if ev >= 1.10: st.success(f"VERDICT : SAFE (Value: {ev:.2f})")
        elif ev >= 0.98: st.warning(f"VERDICT : MID (Value: {ev:.2f})")
        else: st.error(f"VERDICT : ENLÈVE (Value: {ev:.2f})")

    st.markdown("<div class='autobet-card'>", unsafe_allow_html=True)
    st.subheader("🤖 MODE AUTOBET")
    atb1, atb2, atb3, atb4 = st.columns(4)
    bankroll = atb1.number_input("Capital (€)", value=1000)
    c_h, c_n, c_a = atb2.number_input(f"Cote {d['t_h']}", value=2.0), atb3.number_input("Cote NUL", value=3.0), atb4.number_input(f"Cote {d['t_a']}", value=3.0)

    options = [
        {"name": d['t_h'], "prob": d['p_h'], "cote": c_h},
        {"name": "Nul", "prob": d['p_n'], "cote": c_n},
        {"name": d['t_a'], "prob": d['p_a'], "cote": c_a},
        {"name": f"{d['t_h']} ou Nul", "prob": d['p_h'] + d['p_n'], "cote": 1/((1/c_h)+(1/c_n))*0.95},
        {"name": f"Nul ou {d['t_a']}", "prob": d['p_n'] + d['p_a'], "cote": 1/((1/c_n)+(1/c_a))*0.95},
        {"name": f"{d['t_h']} ou {d['t_a']}", "prob": d['p_h'] + d['p_a'], "cote": 1/((1/c_h)+(1/c_a))*0.95}
    ]

    best = None
    max_ev = 0
    for o in options:
        o['ev'] = o['prob'] * o['cote']
        if o['ev'] > 1.05 and o['ev'] > max_ev:
            max_ev = o['ev']
            best = o

    if best:
        b = best['cote'] - 1
        kelly = max(0, ((b * best['prob']) - (1 - best['prob'])) / b) * 0.15
        mise = min(bankroll * kelly, bankroll * 0.05)
        st.write(f"### ✅ CONSEIL : PARIER SUR **{best['name'].upper()}**")
        st.write(f"Mise : **{mise:.2f}€** | Confiance : **{best['prob']*100:.1f}%**")
    else: st.write("### ❌ AUCUN PARI VALUE DETECTÉ")
    st.markdown("</div>", unsafe_allow_html=True)

    st.divider()
    st.subheader("SCORES PROBABLES")
    idx = np.unravel_index(np.argsort(d['matrix'].ravel())[-3:][::-1], d['matrix'].shape)
    s1, s2, s3 = st.columns(3)
    for i in range(3):
        with [s1, s2, s3][i]:
            st.write(f"**TOP {i+1}**")
            st.write(f"{idx[0][i]} - {idx[1][i]} ({d['matrix'][idx[0][i], idx[1][i]]*100:.1f}%)")
