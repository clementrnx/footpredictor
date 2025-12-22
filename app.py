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
    .stApp > div:first-child { background-color: rgba(0, 0, 0, 0.88); }
    
    h1, h2, h3, p, span, label { color: #FFD700 !important; font-family: 'Monospace', sans-serif; letter-spacing: 2px; }

    /* BOUTON GLASS LONG */
    div.stButton > button {
        background: rgba(255, 215, 0, 0.03) !important;
        backdrop-filter: blur(25px) !important;
        -webkit-backdrop-filter: blur(25px) !important;
        border: 1px solid rgba(255, 215, 0, 0.2) !important;
        color: #FFD700 !important;
        border-radius: 15px !important;
        height: 70px !important;
        width: 100% !important;
        font-weight: 200 !important;
        text-transform: uppercase !important;
        letter-spacing: 12px !important;
        transition: 0.6s all ease-in-out;
        margin-top: 20px;
    }
    
    div.stButton > button:hover { 
        background: rgba(255, 215, 0, 0.1) !important;
        border: 1px solid rgba(255, 215, 0, 0.6) !important;
        letter-spacing: 16px !important;
        box-shadow: 0 0 40px rgba(255, 215, 0, 0.15);
    }

    /* CHAMPS DE SAISIE GLASSMOPHISM TOTAL */
    div[data-baseweb="select"], div[data-baseweb="input"], .stNumberInput input, .stSelectbox div {
        background-color: rgba(255, 255, 255, 0.05) !important;
        backdrop-filter: blur(12px) !important;
        -webkit-backdrop-filter: blur(12px) !important;
        border: 0.5px solid rgba(255, 215, 0, 0.15) !important;
        border-radius: 10px !important;
        color: #FFD700 !important;
    }

    /* Suppression des bordures par défaut de Streamlit */
    div[data-baseweb="base-input"] {
        background-color: transparent !important;
        border: none !important;
    }

    .verdict-text {
        font-size: 26px;
        font-weight: 900;
        text-align: center;
        padding: 30px;
        letter-spacing: 6px;
        text-transform: uppercase;
        border-top: 1px solid rgba(255, 215, 0, 0.1);
        border-bottom: 1px solid rgba(255, 215, 0, 0.1);
        margin: 15px 0;
    }

    .bet-card {
        background: rgba(255, 255, 255, 0.02);
        padding: 30px;
        border-radius: 20px;
        border: 1px solid rgba(255, 215, 0, 0.05);
        margin-bottom: 40px;
    }

    .footer {
        text-align: center;
        padding: 50px 0 20px 0;
        color: rgba(255, 215, 0, 0.6);
        font-family: 'Monospace', sans-serif;
        font-size: 14px;
        letter-spacing: 3px;
    }
    .footer a {
        color: #FFD700 !important;
        text-decoration: none;
        font-weight: bold;
        border: 1px solid rgba(255, 215, 0, 0.2);
        padding: 8px 15px;
        border-radius: 5px;
        transition: 0.3s;
    }
    .footer a:hover {
        background: rgba(255, 215, 0, 0.1);
        box-shadow: 0 0 15px rgba(255, 215, 0, 0.2);
    }
    </style>
""", unsafe_allow_html=True)

# Configuration API
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
l_name = st.selectbox("CHOISIR LA LIGUE", list(leagues.keys()))
l_id = leagues[l_name]

teams_res = get_api("teams", {"league": l_id, "season": SEASON})
teams = {t['team']['name']: t['team']['id'] for t in teams_res}

if teams:
    c1, c2 = st.columns(2)
    t_h = c1.selectbox("DOMICILE", sorted(teams.keys()))
    t_a = c2.selectbox("EXTÉRIEUR", sorted(teams.keys()), index=1)

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
    st.write("---")
    
    m1, m2, m3 = st.columns(3)
    m1.metric(d['t_h'], f"{d['p_h']*100:.1f}%")
    m2.metric("NUL", f"{d['p_n']*100:.1f}%")
    m3.metric(d['t_a'], f"{d['p_a']*100:.1f}%")

    st.subheader("🤖 MODE BET")
    st.markdown("<div class='bet-card'>", unsafe_allow_html=True)
    
    b_c1, b_c2, b_c3, b_c4 = st.columns(4)
    bankroll = b_c1.number_input("CAPITAL TOTAL (€)", value=100.0)
    c_h = b_c2.number_input(f"COTE {d['t_h']}", value=2.0)
    c_n = b_c3.number_input("COTE NUL", value=3.0)
    c_a = b_c4.number_input(f"COTE {d['t_a']}", value=3.0)

    dc_1, dc_2, dc_3 = st.columns(3)
    c_hn = dc_1.number_input(f"COTE {d['t_h']} / NUL", value=1.30)
    c_na = dc_2.number_input(f"COTE NUL / {d['t_a']}", value=1.30)
    c_ha = dc_3.number_input(f"COTE {d['t_h']} / {d['t_a']}", value=1.30)

    opts = [
        {"n": d['t_h'], "p": d['p_h'], "c": c_h},
        {"n": "NUL", "p": d['p_n'], "c": c_n},
        {"n": d['t_a'], "p": d['p_a'], "c": c_a},
        {"n": f"{d['t_h']} OU NUL", "p": d['p_h'] + d['p_n'], "c": c_hn},
        {"n": f"NUL OU {d['t_a']}", "p": d['p_n'] + d['p_a'], "c": c_na},
        {"n": f"{d['t_h']} OU {d['t_a']}", "p": d['p_h'] + d['p_a'], "c": c_ha}
    ]

    best_o = max(opts, key=lambda x: x['p'] * x['c'])
    if best_o['p'] * best_o['c'] > 1.02:
        b_val = best_o['c'] - 1
        k_val = ((b_val * best_o['p']) - (1 - best_o['p'])) / b_val if b_val > 0 else 0
        m_finale = bankroll * k_val
        m_finale = max(bankroll * 0.30, m_finale) 
        m_finale = min(m_finale, bankroll * 1.00) 
        
        st.markdown(f"<div class='verdict-text'>IA RECOMMANDE : {best_o['n']} | MISE : {m_finale:.2f}€</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='verdict-text'>AUCUN VALUE DÉTECTÉ</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.subheader("🔍 AUDIT DU TICKET")
    aud1, aud2 = st.columns(2)
    aud_choix = aud1.selectbox("VOTRE PARI", [d['t_h'], "Nul", d['t_a'], f"{d['t_h']} ou Nul", f"Nul ou {d['t_a']}", f"{d['t_h']} ou {d['t_a']}"])
    aud_cote = aud2.number_input("VOTRE COTE", value=1.50)

    p_audit = d['p_h'] if aud_choix == d['t_h'] else (d['p_n'] if aud_choix == "Nul" else d['p_a'])
    if "ou Nul" in aud_choix and d['t_h'] in aud_choix: p_audit = d['p_h'] + d['p_n']
    elif "Nul ou" in aud_choix: p_audit = d['p_n'] + d['p_a']
    elif "ou" in aud_choix: p_audit = d['p_h'] + d['p_a']
    
    audit_val = p_audit * aud_cote
    stat = "SAFE" if audit_val >= 1.10 else ("MID" if audit_val >= 0.98 else "DANGEREUX")
    st.markdown(f"<div class='verdict-text'>AUDIT : {stat} (EV: {audit_val:.2f})</div>", unsafe_allow_html=True)

    st.subheader("SCORES PROBABLES")
    idx = np.unravel_index(np.argsort(d['matrix'].ravel())[-3:][::-1], d['matrix'].shape)
    sc1, sc2, sc3 = st.columns(3)
    for i in range(3):
        with [sc1, sc2, sc3][i]: st.write(f"**{idx[0][i]} - {idx[1][i]}** ({d['matrix'][idx[0][i], idx[1][i]]*100:.1f}%)")

# --- FOOTER GITHUB ---
st.markdown("""
    <div class='footer'>
        DÉVELOPPÉ PAR ITROZ | 
        <a href='https://github.com/clementrnx' target='_blank'>GITHUB SOURCE</a>
    </div>
""", unsafe_allow_html=True)
