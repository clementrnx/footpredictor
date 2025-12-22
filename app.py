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
        border: 1px solid rgba(255, 215, 0, 0.4) !important;
        color: #FFD700 !important;
        border-radius: 10px !important;
        height: 60px !important;
        width: 100% !important;
        font-weight: 900 !important;
        text-transform: uppercase !important;
    }
    
    div.stButton > button:hover { border: 1px solid #FFD700 !important; box-shadow: 0 0 20px rgba(255, 215, 0, 0.2); }

    div[data-baseweb="select"], div[data-baseweb="input"], input { background: transparent !important; border: none !important; }
    div[data-baseweb="select"] > div, div[data-baseweb="input"] > div {
        background: transparent !important; border-bottom: 2px solid #FFD700 !important; border-radius: 0px !important;
    }

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
    st.subheader("🤖 MODE AUTOBET")
    st.markdown("<div class='autobet-card'>", unsafe_allow_html=True)
    
    col_bank, col_c1, col_cn, col_c2 = st.columns(4)
    bankroll = col_bank.number_input("Capital (€)", value=1000)
    c_h = col_c1.number_input(f"Cote {d['t_h']}", value=2.0)
    c_n = col_cn.number_input("Cote NUL", value=3.0)
    c_a = col_c2.number_input(f"Cote {d['t_a']}", value=3.0)

    # Calcul des options de paris incluant les Doubles Chances
    options = [
        {"name": d['t_h'], "prob": d['p_h'], "cote": c_h},
        {"name": "Nul", "prob": d['p_n'], "cote": c_n},
        {"name": d['t_a'], "prob": d['p_a'], "cote": c_a},
        {"name": f"{d['t_h']} ou Nul", "prob": d['p_h'] + d['p_n'], "cote": 1 / ( (1/c_h) + (1/c_n) ) * 0.9}, # Approximation safe
        {"name": f"Nul ou {d['t_a']}", "prob": d['p_n'] + d['p_a'], "cote": 1 / ( (1/c_n) + (1/c_a) ) * 0.9},
        {"name": f"{d['t_h']} ou {d['t_a']}", "prob": d['p_h'] + d['p_a'], "cote": 1 / ( (1/c_h) + (1/c_a) ) * 0.9}
    ]

    # Calcul de l'EV et de la mise de Kelly pour chaque option
    for opt in options:
        opt['ev'] = (opt['prob'] * opt['cote'])
        # Formule de Kelly : (bp - q) / b  où b = cote-1, p = proba, q = 1-p
        b = opt['cote'] - 1
        if b > 0:
            k = (b * opt['prob'] - (1 - opt['prob'])) / b
            opt['kelly'] = max(0, k * 0.15) # On ne mise que 15% de ce que Kelly suggère (Fractional Kelly) pour être ultra safe
        else: opt['kelly'] = 0

    # Filtrer les Value Bets (EV > 1.0) et prendre le meilleur
    value_bets = [o for o in options if o['ev'] > 1.02]
    
    if value_bets:
        best = max(value_bets, key=lambda x: x['ev'])
        mise_finale = bankroll * best['kelly']
        # Restriction stricte : Jamais plus de 5% de la bankroll
        mise_finale = min(mise_finale, bankroll * 0.05)
        
        st.write(f"### ✅ CONSEIL PRO : PARIER SUR **{best['name'].upper()}**")
        st.write(f"Mise recommandée : **{mise_finale:.2f}€** (Soit { (mise_finale/bankroll)*100 :.1f}% de ton capital)")
        st.write(f"Pourquoi ? L'IA estime la probabilité à **{best['prob']*100:.1f}%** contre une cote de **{best['cote']:.2f}** (Value: {best['ev']:.2f})")
    else:
        st.write("### ❌ AUCUN PARI SAFE")
        st.write("Le risque est trop élevé par rapport aux cotes proposées. Ne parie pas sur ce match.")

    st.markdown("</div>", unsafe_allow_html=True)
