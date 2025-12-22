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

    /* BOUTON STYLE BARRE TRANSLUCIDE */
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
        letter-spacing: 3px !important;
        margin-top: 10px;
        margin-bottom: 10px;
    }
    
    div.stButton > button:hover { 
        background: rgba(255, 215, 0, 0.2) !important;
        border: 1px solid #FFD700 !important; 
    }

    div[data-baseweb="select"], div[data-baseweb="input"], input { background: transparent !important; border: none !important; }
    div[data-baseweb="select"] > div, div[data-baseweb="input"] > div {
        background: transparent !important; border-bottom: 2px solid #FFD700 !important; border-radius: 0px !important;
    }

    .bet-card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(20px);
        border: 2px solid #FFD700;
        padding: 30px;
        border-radius: 20px;
        margin-top: 20px;
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
    r3.metric(d['t_a'], f"{d['p_a']*1
