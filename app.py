import streamlit as st
import requests
import numpy as np
from scipy.stats import poisson

# --- CONFIGURATION ET STYLE PREMIUM ---
st.set_page_config(page_title="iTrOz Predictor Pro", layout="wide")

st.markdown("""
    <style>
    /* Import de police plus moderne */
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;500;800&display=swap');

    .stApp {
        background: #050505;
        background-image: radial-gradient(circle at 50% -20%, #1a1a00 0%, #050505 80%);
    }

    /* Harmonisation Totale */
    * {
        font-family: 'JetBrains Mono', monospace !important;
        color: #FFD700 !important;
    }

    /* Suppression des containers blancs de Streamlit */
    .stApp > div:first-child { background-color: transparent !important; }
    [data-testid="stHeader"] { background: transparent !important; }
    
    /* Blocs Glassmorphism Harmonisés */
    div.stButton > button, 
    div[data-baseweb="select"], 
    div[data-baseweb="input"], 
    .stNumberInput input, 
    .stSelectbox div,
    .stTextInput input,
    div[data-testid="stMetric"] {
        background: rgba(255, 215, 0, 0.02) !important;
        backdrop-filter: blur(20px) !important;
        border: 1px solid rgba(255, 215, 0, 0.1) !important;
        border-radius: 4px !important; /* Bordures plus carrées pour le look pro */
        transition: 0.3s all ease;
    }

    /* Hover & Focus */
    div.stButton > button:hover, .stNumberInput input:focus {
        background: rgba(255, 215, 0, 0.08) !important;
        border: 1px solid rgba(255, 215, 0, 0.5) !important;
        box-shadow: 0 0 15px rgba(255, 215, 0, 0.1);
    }

    /* Verdict & Cards */
    .verdict-box {
        padding: 30px;
        border: 1px solid rgba(255, 215, 0, 0.2);
        background: linear-gradient(90deg, rgba(255,215,0,0.05), transparent);
        text-align: center;
        margin: 20px 0;
        text-transform: uppercase;
        letter-spacing: 3px;
    }

    /* Cacher les labels Streamlit pour épurer */
    label { font-size: 0.8rem !important; opacity: 0.6; }

    /* Custom Metric Style */
    div[data-testid="stMetricValue"] { font-size: 2rem !important; font-weight: 800 !important; }
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

@st.cache_data(ttl=3600)
def get_league_context():
    return {'avg_home': 1.55, 'avg_away': 1.25, 'avg_home_conceded': 1.25, 'avg_away_conceded': 1.55}

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
st.title("ITROZ / PREDICTOR")

# Sélection
leagues = {"La Liga": 140, "Champions League": 2, "Premier League": 39, "Serie A": 135, "Ligue 1": 61}
l_name = st.selectbox("LIGUE", list(leagues.keys()))

teams_res = get_api("teams", {"league": leagues[l_name], "season": SEASON})
teams = {t['team']['name']: t['team']['id'] for t in teams_res}

if teams:
    c1, c2 = st.columns(2)
    t_h = c1.selectbox("DOMICILE", sorted(teams.keys()), index=0)
    t_a = c2.selectbox("EXTÉRIEUR", sorted(teams.keys()), index=1)

    if st.button("EXECUTE ANALYSIS", use_container_width=True):
        with st.spinner("CALCULATING..."):
            ctx = get_league_context()
            s_h, s_a = get_stats(teams[t_h], leagues[l_name], SEASON, True), get_stats(teams[t_a], leagues[l_name], SEASON, False)
            if s_h and s_a:
                lh = ctx['avg_home'] * (s_h['f'] / ctx['avg_home']) * (s_a['a'] / ctx['avg_home_conceded'])
                la = ctx['avg_away'] * (s_a['f'] / ctx['avg_away']) * (s_h['a'] / ctx['avg_home_conceded'])
                matrix = np.zeros((7, 7))
                for x in range(7):
                    for y in range(7):
                        matrix[x, y] = poisson.pmf(x, lh) * poisson.pmf(y, la)
                matrix /= matrix.sum()
                st.session_state.data = {'ph': np.sum(np.tril(matrix, -1)), 'pn': np.sum(np.diag(matrix)), 'pa': np.sum(np.triu(matrix, 1)), 'matrix': matrix, 'th': t_h, 'ta': t_a, 'lh': lh, 'la': la}
                st.session_state.done = True

if st.session_state.get('done'):
    d = st.session_state.data
    st.write("---")
    
    # Probabilités metrics
    m1, m2, m3 = st.columns(3)
    m1.metric(d['th'], f"{d['ph']*100:.1f}%")
    m2.metric("NUL", f"{d['pn']*100:.1f}%")
    m3.metric(d['ta'], f"{d['pa']*100:.1f}%")

    # Modes
    st.write("### RISK PROFILE")
    r1, r2, r3 = st.columns(3)
    if 'mode' not in st.session_state: st.session_state.mode = "SAFE"
    if r1.button("🛡️ SAFE", use_container_width=True): st.session_state.mode = "SAFE"
    if r2.button("⚖️ MID", use_container_width=True): st.session_state.mode = "MID"
    if r3.button("🔥 JOUEUR", use_container_width=True): st.session_state.mode = "JOUEUR"

    conf = {"SAFE": (1.05, 0.25, 0.05), "MID": (1.02, 0.5, 0.15), "JOUEUR": (1.001, 1.0, 0.4)}[st.session_state.mode]

    # Bet Section
    st.markdown("<div class='verdict-box'>", unsafe_allow_html=True)
    b1, b2, b3, b4 = st.columns(4)
    bank = b1.number_input("BANKROLL", value=1000.0)
    ch = b2.number_input(f"COTE {d['th']}", value=2.0)
    cn = b3.number_input("COTE NUL", value=3.0)
    ca = b4.number_input(f"COTE {d['ta']}", value=3.0)

    opts = [{"n": d['th'], "p": d['ph'], "c": ch}, {"n": "NUL", "p": d['pn'], "c": cn}, {"n": "AWAY", "p": d['pa'], "c": ca}]
    valides = [o for o in opts if (o['p'] * o['c']) >= conf[0]]
    
    if valides:
        best = max(valides, key=lambda x: x['p'] * x['c'])
        fk = ((best['c']-1)*best['p'] - (1-best['p'])) / (best['c']-1)
        mise = min(bank * fk * conf[1], bank * conf[2])
        st.write(f"## {st.session_state.mode} : {best['n']} | MISE : {max(0, mise):.2f}€")
    else:
        st.write("### NO OPPORTUNITY DETECTED")
    st.markdown("</div>", unsafe_allow_html=True)

    # Audit
    st.write("### AUDIT")
    a1, a2, a3 = st.columns(3)
    a_choix = a1.selectbox("PARI", [d['th'], "Nul", d['ta']])
    a_cote = a2.number_input("COTE", value=1.8)
    a_mise = a3.number_input("MISE", value=50.0)
    p_a = d['ph'] if a_choix == d['th'] else (d['pn'] if a_choix == "Nul" else d['pa'])
    st.markdown(f"<div class='verdict-box'>EV: {p_a*a_cote:.2f} | GAIN: {a_mise*a_cote:.2f}€</div>", unsafe_allow_html=True)

    # Scores
    st.write("### PROBABLE SCORES")
    idx = np.unravel_index(np.argsort(d['matrix'].ravel())[-5:][::-1], d['matrix'].shape)
    cols = st.columns(5)
    for i in range(5):
        cols[i].metric(f"{idx[0][i]}-{idx[1][i]}", f"{d['matrix'][idx[0][i], idx[1][i]]*100:.1f}%")

st.markdown("<p style='text-align:center; opacity:0.3; margin-top:50px;'>iTrOz OS v2.5</p>", unsafe_allow_html=True)
