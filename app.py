import streamlit as st
import requests
import numpy as np
from scipy.stats import poisson

st.set_page_config(page_title="iTrOz Predictor - VF", layout="wide")

st.markdown("""
    <style>
    .stApp {
        background-image: url("https://media.giphy.com/media/VZrfUvQjXaGEQy1RSn/giphy.gif");
        background-size: cover;
        background-attachment: fixed;
    }
    .stApp > div:first-child {
        background-color: rgba(0, 0, 0, 0.85);
    }
    
    h1, h2, h3, p, span, label { 
        color: #FFD700 !important; 
        font-family: 'Monospace', sans-serif; 
    }
    
    div.stButton > button {
        background-color: #FFD700 !important;
        border: 2px solid #000000 !important;
        border-radius: 8px !important;
        height: 55px !important;
        width: 100% !important;
    }
    div.stButton > button p {
        color: #000000 !important;
        font-weight: 900 !important;
        font-size: 20px !important;
        text-transform: uppercase !important;
    }

    div[data-testid="stMetric"] {
        background: rgba(0, 0, 0, 0.8) !important;
        border: 1px solid #FFD700 !important;
        border-radius: 10px;
    }
    
    input, select, div[data-baseweb="select"] {
        background-color: #000000 !important;
        color: #FFD700 !important;
        border: 1px solid #FFD700 !important;
    }
    
    .audit-card {
        background: rgba(10, 10, 10, 0.95);
        border: 2px solid #FFD700;
        padding: 25px;
        border-radius: 15px;
        text-align: center;
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

st.title("iTrOz Predictor - VF")

leagues = {"Premier League": 39, "La Liga": 140, "Serie A": 135, "Bundesliga": 78, "Ligue 1": 61}
l_name = st.selectbox("Choisir un championnat", list(leagues.keys()))
l_id = leagues[l_name]

teams_res = get_api("teams", {"league": l_id, "season": SEASON})
teams = {t['team']['name']: t['team']['id'] for t in teams_res}

if teams:
    c1, c2 = st.columns(2)
    t_h = c1.selectbox("Équipe Domicile", sorted(teams.keys()))
    t_a = c2.selectbox("Équipe Extérieur", sorted(teams.keys()), index=1)

    if st.button("Lancer la Prédiction"):
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
            rho = -0.06
            for x in range(7):
                for y in range(7):
                    p = poisson.pmf(x, lh) * poisson.pmf(y, la)
                    if x==0 and y==0: p *= (1 - lh*la*rho)
                    elif x==0 and y==1: p *= (1 + lh*rho)
                    elif x==1 and y==0: p *= (1 + la*rho)
                    elif x==1 and y==1: p *= (1 - rho)
                    matrix[x,y] = p
            matrix /= matrix.sum()

            st.session_state.data = {
                'p_h': np.sum(np.tril(matrix, -1)) * 100,
                'p_n': np.sum(np.diag(matrix)) * 100,
                'p_a': np.sum(np.triu(matrix, 1)) * 100,
                'matrix': matrix, 't_h': t_h, 't_a': t_a
            }
            st.session_state.simulation_done = True

if st.session_state.simulation_done:
    d = st.session_state.data
    st.divider()
    
    res1, res2, res3 = st.columns(3)
    res1.metric(f"Victoire {d['t_h']}", f"{d['p_h']:.1f}%")
    res2.metric("Match Nul", f"{d['p_n']:.1f}%")
    res3.metric(f"Victoire {d['t_a']}", f"{d['p_a']:.1f}%")

    st.subheader("Audit de Pari")
    with st.container():
        a_col1, a_col2, a_col3 = st.columns(3)
        choix = a_col1.selectbox("Type de pari", [
            d['t_h'], "Nul", d['t_a'], 
            f"{d['t_h']} ou Nul (1N)", 
            f"Nul ou {d['t_a']} (N2)", 
            f"{d['t_h']} ou {d['t_a']} (12)"
        ])
        cote = a_col2.number_input("Cote bookmaker", value=1.50, step=0.01)
        mise = a_col3.number_input("Mise (€)", value=10)

        if choix == d['t_h']: prob_ia = d['p_h']
        elif choix == "Nul": prob_ia = d['p_n']
        elif choix == d['t_a']: prob_ia = d['p_a']
        elif "1N" in choix: prob_ia = d['p_h'] + d['p_n']
        elif "N2" in choix: prob_ia = d['p_n'] + d['p_a']
        elif "12" in choix: prob_ia = d['p_h'] + d['p_a']
        
        prob_ia /= 100
        ev = prob_ia * cote

        st.markdown("<div class='audit-card'>", unsafe_allow_html=True)
        if ev >= 1.10:
            st.write("<h2 style='color:#00FF00 !important;'>VERDICT : SAFE</h2>", unsafe_allow_html=True)
        elif ev >= 0.98:
            st.write("<h2 style='color:#FFD700 !important;'>VERDICT : MID</h2>", unsafe_allow_html=True)
        else:
            st.write("<h2 style='color:#FF4B4B !important;'>VERDICT : ENLÈVE</h2>", unsafe_allow_html=True)
        
        st.write(f"Probabilité combinée : **{prob_ia*100:.1f}%** | Rentabilité estimée : **{((ev-1)*100):.1f}%**")
        st.markdown("</div>", unsafe_allow_html=True)

    st.subheader("Scores les plus probables")
    idx = np.unravel_index(np.argsort(d['matrix'].ravel())[-3:][::-1], d['matrix'].shape)
    sc_col1, sc_col2, sc_col3 = st.columns(3)
    cols = [sc_col1, sc_col2, sc_col3]
    for i in range(3):
        cols[i].write(f"**Top {i+1}**")
        cols[i].write(f"Score : {idx[0][i]} - {idx[1][i]}")
        cols[i].write(f"({d['matrix'][idx[0][i], idx[1][i]]*100:.1f}%)")
