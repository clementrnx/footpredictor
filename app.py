import streamlit as st
import requests
import numpy as np
from scipy.stats import poisson
from datetime import datetime, timedelta

# --- CONFIGURATION ET STYLE ---
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
    
    h1, h2, h3, p, span, label { color: #FFD700 !important; font-family: 'Monospace', sans-serif; letter-spacing: 2px; }

    div.stButton > button {
        background: rgba(255, 215, 0, 0.05) !important;
        border: 1px solid rgba(255, 215, 0, 0.2) !important;
        color: #FFD700 !important;
        border-radius: 10px !important;
        transition: 0.4s all;
    }
    
    div.stButton > button:hover { 
        border: 1px solid rgba(255, 215, 0, 0.6) !important;
        box-shadow: 0 0 20px rgba(255, 215, 0, 0.2);
    }

    .verdict-text {
        font-size: 26px; font-weight: 900; text-align: center; padding: 30px;
        letter-spacing: 6px; text-transform: uppercase;
        border-top: 1px solid rgba(255, 215, 0, 0.1);
        border-bottom: 1px solid rgba(255, 215, 0, 0.1);
        margin: 15px 0;
    }

    .bet-card {
        background: rgba(255, 255, 255, 0.02);
        padding: 30px; border-radius: 20px;
        border: 1px solid rgba(255, 215, 0, 0.1);
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

@st.cache_data(ttl=3600)
def get_league_context(league_id, season):
    return {'avg_home': 1.5, 'avg_away': 1.2, 'avg_home_conceded': 1.2, 'avg_away_conceded': 1.5, 'avg_total': 2.7}

@st.cache_data(ttl=1800)
def get_weighted_xg_stats(team_id, league_id, season, is_home=True, use_global=False):
    params = {"team": team_id, "season": season, "last": 15} if use_global else {"team": team_id, "league": league_id, "season": season, "last": 10}
    fixtures = get_api("fixtures", params)
    if not fixtures: return None
    
    total_w, xg_f, xg_a, count = 0, 0, 0, 0
    for idx, match in enumerate(fixtures):
        if match['fixture']['status']['short'] != 'FT': continue
        weight, home = 0.9 ** idx, match['teams']['home']['id'] == team_id
        if (is_home and not home) or (not is_home and home): continue
        
        xg_f += float(match['teams']['home' if home else 'away'].get('xg') or match['goals']['home' if home else 'away'] or 0) * weight
        xg_a += float(match['teams']['away' if home else 'home'].get('xg') or match['goals']['away' if home else 'home'] or 0) * weight
        total_w += weight
        count += 1
    return {'xg_for': xg_f/total_w, 'xg_against': xg_a/total_w, 'matches_count': count} if total_w > 0 else None

# --- LOGIQUE PRINCIPALE ---
st.title("ITROZ PREDICTOR PRO")

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

    if st.button("Lancer la prédiction", use_container_width=True):
        with st.spinner("Analyse des datas..."):
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
                        if x==0 and y==0: prob *= 0.87
                        matrix[x, y] = prob
                matrix /= matrix.sum()

                st.session_state.data = {
                    'p_h': np.sum(np.tril(matrix, -1)), 'p_n': np.sum(np.diag(matrix)), 'p_a': np.sum(np.triu(matrix, 1)),
                    'matrix': matrix, 't_h': t_h, 't_a': t_a, 'lh': lh, 'la': la, 'mode_calc': "Global" if use_global_stats else l_name
                }
                st.session_state.simulation_done = True

if st.session_state.get('simulation_done'):
    d = st.session_state.data
    st.write("---")
    
    # Métriques principales
    m1, m2, m3 = st.columns(3)
    m1.metric(d['t_h'], f"{d['p_h']*100:.1f}%")
    m2.metric("NUL", f"{d['p_n']*100:.1f}%")
    m3.metric(d['t_a'], f"{d['p_a']*100:.1f}%")

    # --- NOUVEAU SÉLECTEUR DE MODE ---
    st.subheader("🤖 CONFIGURATION DU PARI")
    if 'risk_mode' not in st.session_state: st.session_state.risk_mode = "SAFE"
    
    rm1, rm2, rm3 = st.columns(3)
    with rm1: 
        if st.button("🛡️ MODE SAFE", use_container_width=True): st.session_state.risk_mode = "SAFE"
    with rm2: 
        if st.button("⚖️ MODE MID", use_container_width=True): st.session_state.risk_mode = "MID"
    with rm3: 
        if st.button("🔥 MODE JOUEUR", use_container_width=True): st.session_state.risk_mode = "JOUEUR"

    conf_map = {
        "SAFE":   {"seuil": 1.05, "kelly": 0.25, "max": 0.05, "color": "#00FFCC"},
        "MID":    {"seuil": 1.02, "kelly": 0.50, "max": 0.15, "color": "#FFD700"},
        "JOUEUR": {"seuil": 1.001, "kelly": 1.0, "max": 0.40, "color": "#FF3131"}
    }
    c_active = conf_map[st.session_state.risk_mode]

    st.markdown(f"<div class='bet-card'>", unsafe_allow_html=True)
    b_c1, b_c2, b_c3, b_c4 = st.columns(4)
    bankroll = b_c1.number_input("CAPITAL TOTAL (€)", value=100.0)
    c_h = b_c2.number_input(f"COTE {d['t_h']}", value=2.0)
    c_n = b_c3.number_input("COTE NUL", value=3.0)
    c_a = b_c4.number_input(f"COTE {d['t_a']}", value=3.0)

    opts = [
        {"n": d['t_h'], "p": d['p_h'], "c": c_h},
        {"n": "NUL", "p": d['p_n'], "c": c_n},
        {"n": d['t_a'], "p": d['p_a'], "c": c_a}
    ]
    
    # Filtrage selon le mode
    valides = [o for o in opts if (o['p'] * o['c']) >= c_active['seuil']]
    
    if valides:
        best_o = max(valides, key=lambda x: x['p'] * x['c'])
        f_k = ((best_o['c']-1)*best_o['p'] - (1-best_o['p'])) / (best_o['c']-1)
        m_finale = min(bankroll * f_k * c_active['kelly'], bankroll * c_active['max'])
        
        st.markdown(f"<div class='verdict-text' style='color:{c_active['color']} !important;'>MODE {st.session_state.risk_mode} : {best_o['n']} | MISE : {max(0, m_finale):.2f}€</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='verdict-text'>AUCUN VALUE DÉTECTÉ</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Audit & Scores (Interface d'origine)
    st.subheader("🔍 AUDIT DU TICKET")
    aud1, aud2 = st.columns(2)
    aud_choix = aud1.selectbox("VOTRE PARI", [d['t_h'], "Nul", d['t_a']])
    aud_cote = aud2.number_input("VOTRE COTE", value=1.50)
    p_audit = d['p_h'] if aud_choix == d['t_h'] else (d['p_n'] if aud_choix == "Nul" else d['p_a'])
    st.markdown(f"<div class='verdict-text'>AUDIT : {'SAFE' if p_audit*aud_cote >= 1.10 else 'MID'} (EV: {p_audit*aud_cote:.2f})</div>", unsafe_allow_html=True)

    st.subheader("SCORES PROBABLES")
    idx = np.unravel_index(np.argsort(d['matrix'].ravel())[-5:][::-1], d['matrix'].shape)
    score_cols = st.columns(5)
    for i in range(5):
        with score_cols[i]: 
            st.write(f"**{idx[0][i]} - {idx[1][i]}**")
            st.write(f"{d['matrix'][idx[0][i], idx[1][i]]*100:.1f}%")
    
    with st.expander("📊 DÉTAILS TECHNIQUES"):
        st.write(f"Mode: {d['mode_calc']} | Lambda H: {d['lh']:.2f} | Lambda A: {d['la']:.2f}")

st.markdown("<div style='text-align:center; padding:20px; opacity:0.6;'>DÉVELOPPÉ PAR ITROZ</div>", unsafe_allow_html=True)
