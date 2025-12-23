import streamlit as st
import requests
import numpy as np
from scipy.stats import poisson

# --- CONFIGURATION ET STYLE ORIGINAL HARMONISÉ ---
st.set_page_config(page_title="iTrOz Predictor Pro", layout="wide")

st.markdown("""
    <style>
    @keyframes subtleDistort {
        0% { transform: scale(1.0); filter: hue-rotate(0deg) brightness(1); }
        50% { transform: scale(1.02) contrast(1.1); filter: hue-rotate(2deg) brightness(1.1); }
        100% { transform: scale(1.0); filter: hue-rotate(0deg) brightness(1); }
    }

    .stApp {
        background-image: url("https://media.giphy.com/media/VZrfUvQjXaGEQy1RSn/giphy.gif");
        background-size: cover;
        background-attachment: fixed;
        animation: subtleDistort 10s infinite ease-in-out;
    }

    .stApp > div:first-child { background-color: rgba(0, 0, 0, 0.88); position: relative; z-index: 2; }
    
    /* Harmonisation des textes */
    h1, h2, h3, p, span, label { color: #FFD700 !important; font-family: 'Monospace', sans-serif; letter-spacing: 2px; }

    /* Harmonisation Glassmorphism + Bords Arrondis */
    div.stButton > button, 
    div[data-baseweb="select"], 
    div[data-baseweb="input"], 
    .stNumberInput input, 
    .stSelectbox div,
    .stTextInput input {
        background: rgba(255, 215, 0, 0.05) !important;
        backdrop-filter: blur(20px) !important;
        -webkit-backdrop-filter: blur(20px) !important;
        border: 1px solid rgba(255, 215, 0, 0.2) !important;
        border-radius: 30px !important; /* Boutons et champs bien arrondis */
        color: #FFD700 !important;
        transition: 0.4s all ease-in-out;
        padding: 10px 20px !important;
    }
    
    /* Effet Hover Harmonisé */
    div.stButton > button:hover, .stNumberInput input:focus { 
        background: rgba(255, 215, 0, 0.12) !important;
        border: 1px solid rgba(255, 215, 0, 0.6) !important;
        box-shadow: 0 0 25px rgba(255, 215, 0, 0.2);
        transform: translateY(-2px);
    }

    .verdict-box {
        background: rgba(255, 215, 0, 0.03);
        backdrop-filter: blur(15px);
        border: 1px solid rgba(255, 215, 0, 0.2);
        padding: 25px;
        border-radius: 25px;
        text-align: center;
        margin: 20px 0;
        text-transform: uppercase;
    }

    /* GitHub Button Style */
    .github-btn {
        display: inline-block;
        padding: 12px 25px;
        border-radius: 30px;
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 215, 0, 0.3);
        color: #FFD700;
        text-decoration: none;
        font-weight: bold;
        transition: 0.3s;
    }
    .github-btn:hover {
        background: rgba(255, 215, 0, 0.1);
        border-color: #FFD700;
    }
    </style>
""", unsafe_allow_html=True)

# API Config
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

@st.cache_data(ttl=1800)
def get_stats(team_id, league_id, season, is_home=True):
    fixtures = get_api("fixtures", {"team": team_id, "league": league_id, "season": season, "last": 10})
    if not fixtures: return None
    tw, xgf, xga = 0, 0, 0
    for i, m in enumerate(fixtures):
        if m['fixture']['status']['short'] != 'FT': continue
        w, home = 0.9 ** i, m['teams']['home']['id'] == team_id
        if (is_home and not home) or (not is_home and home): continue
        xgf += float(m['teams']['home' if home else 'away'].get('xg') or m['goals']['home' if home else 'away'] or 0) * w
        xga += float(m['teams']['away' if home else 'home'].get('xg') or m['goals']['away' if home else 'home'] or 0) * w
        tw += w
    return {'f': xgf/tw, 'a': xga/tw} if tw > 0 else None

# --- UI ---
st.title("ITROZ / PREDICTOR PRO")

leagues = {"La Liga": 140, "Champions League": 2, "Premier League": 39, "Serie A": 135, "Ligue 1": 61}
l_name = st.selectbox("LIGUE", list(leagues.keys()))

teams_res = get_api("teams", {"league": leagues[l_name], "season": SEASON})
teams = {t['team']['name']: t['team']['id'] for t in teams_res}

if teams:
    c1, c2 = st.columns(2)
    t_h = c1.selectbox("DOMICILE", sorted(teams.keys()), index=0)
    t_a = c2.selectbox("EXTÉRIEUR", sorted(teams.keys()), index=1)

    if st.button("LANCER L'ANALYSE", use_container_width=True):
        with st.spinner("TRAITEMENT DES DONNÉES..."):
            s_h, s_a = get_stats(teams[t_h], leagues[l_name], SEASON, True), get_stats(teams[t_a], leagues[l_name], SEASON, False)
            if s_h and s_a:
                lh, la = (s_h['f'] * s_a['a'])**0.5 + 0.5, (s_a['f'] * s_h['a'])**0.5 + 0.3
                matrix = np.zeros((7, 7))
                for x in range(7):
                    for y in range(7):
                        matrix[x, y] = poisson.pmf(x, lh) * poisson.pmf(y, la)
                matrix /= matrix.sum()
                st.session_state.data = {'ph': np.sum(np.tril(matrix, -1)), 'pn': np.sum(np.diag(matrix)), 'pa': np.sum(np.triu(matrix, 1)), 'matrix': matrix, 'th': t_h, 'ta': t_a}
                st.session_state.done = True

if st.session_state.get('done'):
    d = st.session_state.data
    st.write("---")
    
    # Probabilités
    m1, m2, m3 = st.columns(3)
    m1.metric(d['th'], f"{d['ph']*100:.1f}%")
    m2.metric("NUL", f"{d['pn']*100:.1f}%")
    m3.metric(d['ta'], f"{d['pa']*100:.1f}%")

    st.subheader("🤖 PROFIL DE RISQUE")
    if 'mode' not in st.session_state: st.session_state.mode = "SAFE"
    r1, r2, r3 = st.columns(3)
    if r1.button("🛡️ SAFE", use_container_width=True): st.session_state.mode = "SAFE"
    if r2.button("⚖️ MID", use_container_width=True): st.session_state.mode = "MID"
    if r3.button("🔥 JOUEUR", use_container_width=True): st.session_state.mode = "JOUEUR"

    conf = {"SAFE": (1.05, 0.25, 0.05), "MID": (1.02, 0.5, 0.15), "JOUEUR": (1.001, 1.0, 0.4)}[st.session_state.mode]

    # Section Betting
    st.markdown("<div class='verdict-box'>", unsafe_allow_html=True)
    b1, b2, b3, b4 = st.columns(4)
    bank = b1.number_input("BANKROLL", value=100.0)
    ch, cn, ca = b2.number_input(f"COTE {d['th']}", value=2.0), b3.number_input("COTE NUL", value=3.0), b4.number_input(f"COTE {d['ta']}", value=3.0)

    opts = [{"n": d['th'], "p": d['ph'], "c": ch}, {"n": "NUL", "p": d['pn'], "c": cn}, {"n": d['ta'], "p": d['pa'], "c": ca}]
    valides = [o for o in opts if (o['p'] * o['c']) >= conf[0]]
    
    if valides:
        best = max(valides, key=lambda x: x['p'] * x['c'])
        fk = ((best['c']-1)*best['p'] - (1-best['p'])) / (best['c']-1)
        mise = min(bank * fk * conf[1], bank * conf[2])
        st.write(f"### {st.session_state.mode} : {best['n']} | MISE : {max(0, mise):.2f}€")
    else:
        st.write("### AUCUN VALUE DÉTECTÉ")
    st.markdown("</div>", unsafe_allow_html=True)

    # Section Audit
    st.subheader("🔍 AUDIT DU TICKET")
    st.markdown("<div class='verdict-box'>", unsafe_allow_html=True)
    a1, a2, a3 = st.columns(3)
    a_choix = a1.selectbox("PARI", [d['th'], "Nul", d['ta']])
    a_cote = a2.number_input("COTE", value=1.5)
    a_mise = a3.number_input("MISE (€)", value=10.0)
    p_a = d['ph'] if a_choix == d['th'] else (d['pn'] if a_choix == "Nul" else d['pa'])
    st.write(f"EV: {p_a*a_cote:.2f} | GAIN POTENTIEL: {a_mise*a_cote:.2f}€")
    st.markdown("</div>", unsafe_allow_html=True)

    # Scores Probables
    st.subheader("🎯 SCORES PROBABLES")
    idx = np.unravel_index(np.argsort(d['matrix'].ravel())[-5:][::-1], d['matrix'].shape)
    cols = st.columns(5)
    for i in range(5):
        cols[i].metric(f"{idx[0][i]}-{idx[1][i]}", f"{d['matrix'][idx[0][i], idx[1][i]]*100:.1f}%")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; padding-bottom: 40px;'>
        <p style='opacity: 0.5;'>iTrOz Predictor v2.5</p>
        <a href='https://github.com/votre-username' class='github-btn' target='_blank'>
            📂 VOIR MON GITHUB
        </a>
    </div>
""", unsafe_allow_html=True)
