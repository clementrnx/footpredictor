import streamlit as st
import requests
import numpy as np
from scipy.stats import poisson
from datetime import datetime, timedelta

# --- CONFIGURATION ET STYLE GLASSMORPHISM ---
st.set_page_config(page_title="iTrOz Predictor", layout="wide")

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

    .stApp > div:first-child { background-color: rgba(0, 0, 0, 0.85); position: relative; z-index: 2; }
    
    h1, h2, h3, p, span, label { color: #FFD700 !important; font-family: 'Monospace', sans-serif; letter-spacing: 2px; }

    /* Boutons et Éléments Glassmorphism */
    div.stButton > button, div[data-baseweb="select"], div[data-baseweb="input"], .stNumberInput input, .stSelectbox div {
        background: rgba(255, 215, 0, 0.03) !important;
        backdrop-filter: blur(15px) !important;
        -webkit-backdrop-filter: blur(15px) !important;
        border: 1px solid rgba(255, 215, 0, 0.15) !important;
        color: #FFD700 !important;
        border-radius: 12px !important;
        transition: 0.4s all ease-in-out;
    }
    
    div.stButton > button {
        height: 60px !important;
        text-transform: uppercase !important;
        letter-spacing: 4px !important;
    }
    
    div.stButton > button:hover { 
        background: rgba(255, 215, 0, 0.1) !important;
        border: 1px solid rgba(255, 215, 0, 0.5) !important;
        box-shadow: 0 0 30px rgba(255, 215, 0, 0.1);
    }

    .verdict-text {
        font-size: 24px; font-weight: 900; text-align: center; padding: 25px;
        letter-spacing: 3px; text-transform: uppercase;
        background: rgba(255, 215, 0, 0.02);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 215, 0, 0.1);
        border-radius: 15px;
        margin: 15px 0;
    }

    .bet-card {
        background: rgba(255, 255, 255, 0.02);
        backdrop-filter: blur(20px);
        padding: 30px; border-radius: 20px;
        border: 1px solid rgba(255, 215, 0, 0.05);
        margin-bottom: 20px;
    }

    /* Suppression de la barre latérale Streamlit par défaut une fois l'app lancée */
    [data-testid="stSidebar"] {
        background-color: rgba(0, 0, 0, 0.7) !important;
        backdrop-filter: blur(10px);
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

@st.cache_data(ttl=3600)
def get_league_context(league_id, season):
    return {'avg_home': 1.55, 'avg_away': 1.25, 'avg_home_conceded': 1.25, 'avg_away_conceded': 1.55, 'avg_total': 2.8}

@st.cache_data(ttl=1800)
def get_weighted_xg_stats(team_id, league_id, season, is_home=True, use_global=False):
    params = {"team": team_id, "season": season, "last": 15} if use_global else {"team": team_id, "league": league_id, "season": season, "last": 10}
    fixtures = get_api("fixtures", params)
    if not fixtures: return None
    
    total_w, xg_f, xg_a = 0, 0, 0
    for idx, match in enumerate(fixtures):
        if match['fixture']['status']['short'] != 'FT': continue
        weight, home = 0.9 ** idx, match['teams']['home']['id'] == team_id
        if (is_home and not home) or (not is_home and home): continue
        xg_f += float(match['teams']['home' if home else 'away'].get('xg') or match['goals']['home' if home else 'away'] or 0) * weight
        xg_a += float(match['teams']['away' if home else 'home'].get('xg') or match['goals']['away' if home else 'home'] or 0) * weight
        total_w += weight
    return {'xg_for': xg_f/total_w, 'xg_against': xg_a/total_w} if total_w > 0 else None

# --- APP ---
st.title("ITROZ PREDICTOR")

col_toggle, col_league = st.columns([1, 3])
with col_toggle:
    use_global_stats = st.toggle("📊 MODE GLOBAL", value=False)

leagues = {"La Liga": 140, "Champions League": 2, "Premier League": 39, "Serie A": 135, "Bundesliga": 78, "Ligue 1": 61}
with col_league:
    l_name = st.selectbox("CHOISIR LA LIGUE", list(leagues.keys()))

teams_res = get_api("teams", {"league": leagues[l_name], "season": SEASON})
teams = {t['team']['name']: t['team']['id'] for t in teams_res}

if teams:
    c1, c2 = st.columns(2)
    t_h = c1.selectbox("DOMICILE", sorted(teams.keys()), index=0)
    t_a = c2.selectbox("EXTÉRIEUR", sorted(teams.keys()), index=1)

    if st.button("LANCER LA PRÉDICTION", use_container_width=True):
        with st.spinner("ANALYSE XG..."):
            ctx = get_league_context(leagues[l_name], SEASON)
            s_h = get_weighted_xg_stats(teams[t_h], leagues[l_name], SEASON, True, use_global_stats)
            s_a = get_weighted_xg_stats(teams[t_a], leagues[l_name], SEASON, False, use_global_stats)
            
            if s_h and s_a:
                lh = ctx['avg_home'] * (s_h['xg_for'] / ctx['avg_home']) * (s_a['xg_against'] / ctx['avg_home_conceded'])
                la = ctx['avg_away'] * (s_a['xg_for'] / ctx['avg_away']) * (s_h['xg_against'] / ctx['avg_away_conceded'])
                matrix = np.zeros((8, 8))
                for x in range(8):
                    for y in range(8):
                        prob = poisson.pmf(x, lh) * poisson.pmf(y, la)
                        if x==0 and y==0: prob *= 0.88
                        matrix[x, y] = prob
                matrix /= matrix.sum()
                st.session_state.data = {'p_h': np.sum(np.tril(matrix, -1)), 'p_n': np.sum(np.diag(matrix)), 'p_a': np.sum(np.triu(matrix, 1)), 'matrix': matrix, 't_h': t_h, 't_a': t_a, 'lh': lh, 'la': la}
                st.session_state.simulation_done = True

if st.session_state.get('simulation_done'):
    d = st.session_state.data
    st.write("---")
    
    m1, m2, m3 = st.columns(3)
    m1.metric(d['t_h'], f"{d['p_h']*100:.1f}%")
    m2.metric("NUL", f"{d['p_n']*100:.1f}%")
    m3.metric(d['t_a'], f"{d['p_a']*100:.1f}%")

    st.subheader("🤖 CONFIGURATION BET")
    if 'risk_mode' not in st.session_state: st.session_state.risk_mode = "SAFE"
    
    rm1, rm2, rm3 = st.columns(3)
    with rm1: 
        if st.button("🛡️ SAFE"): st.session_state.risk_mode = "SAFE"
    with rm2: 
        if st.button("⚖️ MID"): st.session_state.risk_mode = "MID"
    with rm3: 
        if st.button("🔥 JOUEUR"): st.session_state.risk_mode = "JOUEUR"

    conf = {"SAFE": {"s": 1.05, "k": 0.25, "m": 0.05}, "MID": {"s": 1.02, "k": 0.50, "m": 0.15}, "JOUEUR": {"s": 1.001, "k": 1.0, "m": 0.40}}[st.session_state.risk_mode]

    st.markdown("<div class='bet-card'>", unsafe_allow_html=True)
    b_c1, b_c2, b_c3, b_c4 = st.columns(4)
    bankroll = b_c1.number_input("CAPITAL TOTAL (€)", value=100.0)
    c_h = b_c2.number_input(f"COTE {d['t_h']}", value=2.0)
    c_n = b_c3.number_input("COTE NUL", value=3.0)
    c_a = b_c4.number_input(f"COTE {d['t_a']}", value=3.0)

    opts = [{"n": d['t_h'], "p": d['p_h'], "c": c_h}, {"n": "NUL", "p": d['p_n'], "c": c_n}, {"n": d['t_a'], "p": d['p_a'], "c": c_a}]
    valides = [o for o in opts if (o['p'] * o['c']) >= conf['s']]
    
    if valides:
        best = max(valides, key=lambda x: x['p'] * x['x'] if 'x' in x else x['p']*x['c'])
        f_k = ((best['c']-1)*best['p'] - (1-best['p'])) / (best['c']-1)
        mise_rec = min(bankroll * f_k * conf['k'], bankroll * conf['m'])
        st.markdown(f"<div class='verdict-text'>{st.session_state.risk_mode} : {best['n']} | MISE : {max(0, mise_rec):.2f}€</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='verdict-text'>AUCUN VALUE DÉTECTÉ</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # --- SECTION AUDIT MISE À JOUR ---
    st.subheader("🔍 AUDIT DU TICKET")
    st.markdown("<div class='bet-card'>", unsafe_allow_html=True)
    aud_col1, aud_col2, aud_col3 = st.columns(3)
    aud_choix = aud_col1.selectbox("VOTRE PARI", [d['t_h'], "Nul", d['t_a']])
    aud_cote = aud_col2.number_input("VOTRE COTE", value=1.50)
    aud_mise = aud_col3.number_input("VOTRE MISE (€)", value=10.0)
    
    p_aud = d['p_h'] if aud_choix == d['t_h'] else (d['p_n'] if aud_choix == "Nul" else d['p_a'])
    ev = p_aud * aud_cote
    gain_pot = aud_mise * aud_cote
    
    st.markdown(f"""
        <div class='verdict-text'>
            ANALYSE : {'✅ SAFE' if ev >= 1.10 else '⚠️ MID'} | EV: {ev:.2f}<br>
            <span style='font-size: 16px; opacity: 0.8;'>GAIN POTENTIEL : {gain_pot:.2f}€ (Probabilité: {p_aud*100:.1f}%)</span>
        </div>
    """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.subheader("🎯 SCORES PROBABLES")
    idx = np.unravel_index(np.argsort(d['matrix'].ravel())[-5:][::-1], d['matrix'].shape)
    cols = st.columns(5)
    for i in range(5):
        with cols[i]:
            st.markdown(f"""
                <div style='background: rgba(255, 215, 0, 0.03); border: 1px solid rgba(255, 215, 0, 0.1); padding: 15px; border-radius: 10px; text-align: center;'>
                    <span style='font-size: 20px; font-weight: bold;'>{idx[0][i]} - {idx[1][i]}</span><br>
                    <span style='color: #FFD700;'>{d['matrix'][idx[0][i], idx[1][i]]*100:.1f}%</span>
                </div>
            """, unsafe_allow_html=True)

    with st.expander("📊 DÉTAILS TECHNIQUES"):
        st.write(f"Mode: {d.get('mode_calc', 'N/A')} | Lambda H: {d['lh']:.2f} | Lambda A: {d['la']:.2f}")

st.markdown("<div style='text-align:center; padding:20px; opacity:0.6;'>DÉVELOPPÉ PAR ITROZ</div>", unsafe_allow_html=True)
