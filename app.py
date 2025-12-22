import streamlit as st
import requests
import numpy as np
from scipy.stats import poisson

st.set_page_config(page_title="iTrOz Predictor v3", layout="wide")

# --- DESIGN SIGNATURE ITROZ ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@100;200;300;900&display=swap');

    .stApp {
        background-image: url("https://media.giphy.com/media/VZrfUvQjXaGEQy1RSn/giphy.gif");
        background-size: cover;
        background-attachment: fixed;
    }
    .stApp > div:first-child { background-color: rgba(0, 0, 0, 0.92); }

    /* TEXTES & LABELS : PUR MINIMALISME */
    h1, h2, h3, p, span, label, div[data-testid="stWidgetLabel"] p {
        color: #FFD700 !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 200 !important;
        text-transform: uppercase !important;
        letter-spacing: 4px !important;
        background: transparent !important;
    }

    h1 { font-weight: 900 !important; letter-spacing: 15px !important; text-align: center; margin-bottom: 50px; }

    /* CHAMPS DE SAISIE : LAMES DE VERRE */
    div[data-baseweb="select"], div[data-baseweb="input"], .stNumberInput input, .stSelectbox div {
        background-color: rgba(255, 255, 255, 0.02) !important;
        backdrop-filter: blur(20px) !important;
        border: none !important;
        border-bottom: 1px solid rgba(255, 215, 0, 0.2) !important;
        border-radius: 0px !important;
        color: #FFD700 !important;
        padding: 10px 0 !important;
    }

    /* BOUTON D'ACTION : INFINITY STYLE */
    div.stButton > button {
        background: rgba(255, 215, 0, 0.02) !important;
        backdrop-filter: blur(10px) !important;
        border: 1px solid rgba(255, 215, 0, 0.1) !important;
        color: #FFD700 !important;
        border-radius: 0px !important;
        height: 80px !important;
        width: 100% !important;
        font-weight: 100 !important;
        font-size: 18px !important;
        text-transform: uppercase !important;
        letter-spacing: 20px !important;
        transition: 1s cubic-bezier(0.19, 1, 0.22, 1);
        margin-top: 50px;
    }
    
    div.stButton > button:hover { 
        background: rgba(255, 215, 0, 0.1) !important;
        border: 1px solid rgba(255, 215, 0, 0.5) !important;
        letter-spacing: 25px !important;
    }

    /* VERDICT : ÉLÉGANCE EXPLICITE */
    .verdict-container {
        text-align: center;
        padding: 60px 20px;
        margin: 40px 0;
        border-top: 1px solid rgba(255, 215, 0, 0.05);
        border-bottom: 1px solid rgba(255, 215, 0, 0.05);
    }
    .verdict-label { font-size: 14px; opacity: 0.6; margin-bottom: 10px; }
    .verdict-main { font-size: 38px; font-weight: 900; letter-spacing: 12px; }
    .verdict-sub { font-size: 18px; margin-top: 10px; opacity: 0.8; }

    /* CARTES & STRUCTURE */
    .stMetric { background: transparent !important; border: none !important; }
    hr { border: 0; border-top: 1px solid rgba(255, 215, 0, 0.1); margin: 50px 0; }
    </style>
""", unsafe_allow_html=True)

# --- LOGIQUE CORE (LOCK) ---
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

st.write("# iTrOz Predictor")

leagues = {"Champions League": 2, "Premier League": 39, "La Liga": 140, "Serie A": 135, "Bundesliga": 78, "Ligue 1": 61}
l_name = st.selectbox("SÉLECTIONNER LIGUE", list(leagues.keys()))
l_id = leagues[l_name]

teams_res = get_api("teams", {"league": l_id, "season": SEASON})
teams = {t['team']['name']: t['team']['id'] for t in teams_res}

if teams:
    col1, col2 = st.columns(2)
    t_h = col1.selectbox("EQUIPE DOMICILE", sorted(teams.keys()))
    t_a = col2.selectbox("EQUIPE EXTÉRIEUR", sorted(teams.keys()), index=1)

    if st.button("Lancer la prédiction"):
        id_h, id_a = teams[t_h], teams[t_a]
        s_h = get_api("teams/statistics", {"league": l_id, "season": SEASON, "team": id_h})
        s_a = get_api("teams/statistics", {"league": l_id, "season": SEASON, "team": id_a})
        
        if s_h and s_a:
            # Calcul Poisson (Maths Lock)
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
    st.markdown("---")
    
    # --- RÉSULTATS IA (PRIORITAIRE) ---
    st.write("## 🤖 Analyse Prédictive")
    
    b_col1, b_col2, b_col3, b_col4 = st.columns(4)
    bankroll = b_col1.number_input("BANKROLL (€)", value=100.0)
    c_h, c_n, c_a = b_col2.number_input(f"COTE {d['t_h']}", 2.0), b_col3.number_input("COTE NUL", 3.0), b_col4.number_input(f"COTE {d['t_a']}", 3.0)

    dc_1, dc_2, dc_3 = st.columns(3)
    c_hn, c_na, c_ha = dc_1.number_input(f"COTE {d['t_h']}/N", 1.30), dc_2.number_input(f"COTE N/{d['t_a']}", 1.30), dc_3.number_input(f"COTE {d['t_h']}/{d['t_a']}", 1.30)

    opts = [{"n": d['t_h'], "p": d['p_h'], "c": c_h}, {"n": "NUL", "p": d['p_n'], "c": c_n}, {"n": d['t_a'], "p": d['p_a'], "c": c_a},
            {"n": f"{d['t_h']} OU NUL", "p": d['p_h'] + d['p_n'], "c": c_hn}, {"n": f"NUL OU {d['t_a']}", "p": d['p_n'] + d['p_a'], "c": c_na}, {"n": f"{d['t_h']} OU {d['t_a']}", "p": d['p_h'] + d['p_a'], "c": c_ha}]

    best = max(opts, key=lambda x: x['p'] * x['c'])
    st.markdown("<div class='verdict-container'>", unsafe_allow_html=True)
    if best['p'] * best['c'] > 1.02:
        b_val = best['c'] - 1
        k_val = ((b_val * best['p']) - (1 - best['p'])) / b_val if b_val > 0 else 0
        mise = min(max(bankroll * 0.30, bankroll * k_val * 0.5), bankroll * 0.25)
        st.markdown(f"<div class='verdict-label'>RECOMMANDATION SYSTÈME</div><div class='verdict-main'>{best['n']}</div><div class='verdict-sub'>MISE OPTIMALE : {mise:.2f}€</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='verdict-main'>AUCUNE VALUE DÉTECTÉE</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # --- AUDIT ---
    st.write("## 🔍 Audit de Ticket")
    a1, a2 = st.columns(2)
    a_choix = a1.selectbox("VOTRE SÉLECTION", [d['t_h'], "Nul", d['t_a'], f"{d['t_h']} ou Nul", f"Nul ou {d['t_a']}", f"{d['t_h']} ou {d['t_a']}"])
    a_cote = a2.number_input("COTE DU TICKET", value=1.50)

    p_audit = d['p_h'] if a_choix == d['t_h'] else (d['p_n'] if a_choix == "Nul" else d['p_a'])
    if "ou Nul" in a_choix and d['t_h'] in a_choix: p_audit = d['p_h'] + d['p_n']
    elif "Nul ou" in a_choix: p_audit = d['p_n'] + d['p_a']
    elif "ou" in a_choix: p_audit = d['p_h'] + d['p_a']
    
    ev = p_audit * a_cote
    stat = "CONFIRMÉ" if ev >= 1.10 else ("INCERTAIN" if ev >= 0.98 else "RISQUÉ")
    st.markdown(f"<div class='verdict-container'><div class='verdict-label'>STATUT DE L'AUDIT</div><div class='verdict-main'>{stat}</div><div class='verdict-sub'>EXPECTED VALUE : {ev:.2f}</div></div>", unsafe_allow_html=True)

    # --- SCORES ---
    st.write("## 📊 Scores Probables")
    idx = np.unravel_index(np.argsort(d['matrix'].ravel())[-3:][::-1], d['matrix'].shape)
    s1, s2, s3 = st.columns(3)
    for i in range(3):
        with [s1, s2, s3][i]: 
            st.metric(f"SCORE {i+1}", f"{idx[0][i]} - {idx[1][i]}", delta=f"{d['matrix'][idx[0][i], idx[1][i]]*100:.1f}%", delta_color="off")
