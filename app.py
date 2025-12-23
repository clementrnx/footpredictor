import streamlit as st
import requests
import numpy as np
from scipy.stats import poisson
from datetime import datetime

# --- CONFIGURATION INITIALE ---
st.set_page_config(page_title="iTrOz Predictor", layout="wide")

# Initialisation du Session State (Indispensable pour éviter les erreurs AttributeError)
if 'simulation_done' not in st.session_state:
    st.session_state.simulation_done = False
if 'data' not in st.session_state:
    st.session_state.data = {}

# --- STYLE CSS ---
st.markdown("""
    <style>
    @keyframes subtleDistort {
        0% { transform: scale(1.0); filter: hue-rotate(0deg) brightness(1); }
        50% { transform: scale(1.02) contrast(1.1); filter: hue-rotate(2deg) brightness(1.1); }
        100% { transform: scale(1.0); filter: hue-rotate(0deg) brightness(1); }
    }
    .stApp {
        background-color: #0e1117;
        animation: subtleDistort 10s infinite ease-in-out;
    }
    h1, h2, h3, p, span, label { color: #FFD700 !important; font-family: 'Monospace', sans-serif; }
    .verdict-text {
        font-size: 24px; font-weight: 900; text-align: center; padding: 20px;
        border: 2px solid #FFD700; border-radius: 15px; margin: 20px 0;
    }
    .bet-card {
        background: rgba(255, 255, 255, 0.05);
        padding: 25px; border-radius: 15px; border: 1px solid rgba(255, 215, 0, 0.3);
    }
    </style>
""", unsafe_allow_html=True)

# --- CONFIGURATION API ---
API_KEY = st.secrets["MY_API_KEY"]
BASE_URL = "https://v3.football.api-sports.io/"
HEADERS = {'x-apisports-key': API_KEY}
SEASON = 2025

@st.cache_data(ttl=3600)
def get_api(endpoint, params):
    try:
        r = requests.get(f"{BASE_URL}{endpoint}", headers=HEADERS, params=params, timeout=12)
        return r.json().get('response', [])
    except:
        return []

@st.cache_data(ttl=3600)
def get_league_context(league_id, season):
    standings = get_api("standings", {"league": league_id, "season": season})
    if not standings or not standings[0].get('league', {}).get('standings'):
        return {'avg_home': 1.5, 'avg_away': 1.2, 'avg_home_conceded': 1.2, 'avg_away_conceded': 1.5, 'avg_total': 2.7}
    
    t_h_g, t_a_g, t_h_c, t_a_c, t_m = 0, 0, 0, 0, 0
    for team in standings[0]['league']['standings'][0]:
        t_h_g += team['home']['goals']['for']
        t_h_c += team['home']['goals']['against']
        t_a_g += team['away']['goals']['for']
        t_a_c += team['away']['goals']['against']
        t_m += team['home']['played']
    
    return {
        'avg_home': t_h_g / t_m if t_m > 0 else 1.5,
        'avg_away': t_a_g / t_m if t_m > 0 else 1.2,
        'avg_home_conceded': t_h_c / t_m if t_m > 0 else 1.2,
        'avg_away_conceded': t_a_c / t_m if t_m > 0 else 1.5,
        'avg_total': (t_h_g + t_a_g) / (t_m * 2) if t_m > 0 else 2.7
    }

@st.cache_data(ttl=1800)
def get_weighted_xg_stats(team_id, league_id, season, is_home=True, use_global=False):
    params = {"team": team_id, "season": season, "last": 15} if use_global else {"team": team_id, "league": league_id, "season": season, "last": 10}
    fixtures = get_api("fixtures", params)
    if not fixtures: return None
    
    xg_f_w, xg_a_w, total_w, count = 0, 0, 0, 0
    fixtures_sorted = sorted(fixtures, key=lambda x: x['fixture']['date'], reverse=True)
    
    for idx, f in enumerate(fixtures_sorted):
        if f['fixture']['status']['short'] != 'FT': continue
        weight = 0.9 ** idx
        team_is_home = f['teams']['home']['id'] == team_id
        if is_home != team_is_home: continue
        
        xg_f = float(f['teams']['home'].get('xg') or f['goals']['home'] or 0) if team_is_home else float(f['teams']['away'].get('xg') or f['goals']['away'] or 0)
        xg_a = float(f['teams']['away'].get('xg') or f['goals']['away'] or 0) if team_is_home else float(f['teams']['home'].get('xg') or f['goals']['home'] or 0)
            
        xg_f_w += xg_f * weight
        xg_a_w += xg_a * weight
        total_w += weight
        count += 1
    
    return {'xg_for': xg_f_w/total_w, 'xg_against': xg_a_w/total_w, 'matches_count': count} if total_w > 0 else None

# --- INTERFACE ---
st.title("⚽ ITROZ PREDICTOR")

col_toggle, col_aggr, col_league = st.columns([1, 1, 2])
with col_toggle:
    use_global_stats = st.toggle("📊 MODE GLOBAL", value=False)
with col_aggr:
    aggressivity = st.select_slider("🎲 MODE", options=["PRUDENT", "ÉQUILIBRÉ", "JOUEUR", "RISQUÉ"], value="JOUEUR")

leagues = {"La Liga": 140, "Champions League": 2, "Premier League": 39, "Serie A": 135, "Bundesliga": 78, "Ligue 1": 61}
with col_league:
    l_name = st.selectbox("LIGUE", list(leagues.keys()))
l_id = leagues[l_name]

# Récupération des équipes
teams_res = get_api("teams", {"league": l_id, "season": SEASON})
teams = {t['team']['name']: t['team']['id'] for t in teams_res}

if teams:
    sorted_names = sorted(teams.keys())
    c1, c2 = st.columns(2)
    t_h = c1.selectbox("DOMICILE", sorted_names, index=0)
    t_a = c2.selectbox("EXTÉRIEUR", sorted_names, index=1 if len(sorted_names) > 1 else 0)

    if st.button("Lancer la prédiction"):
        with st.spinner("Analyse xG en cours..."):
            id_h, id_a = teams[t_h], teams[t_a]
            l_ctx = get_league_context(l_id, SEASON)
            
            s_h_xg = get_weighted_xg_stats(id_h, l_id, SEASON, True, use_global_stats)
            s_a_xg = get_weighted_xg_stats(id_a, l_id, SEASON, False, use_global_stats)
            
            att_h = s_h_xg['xg_for'] if s_h_xg else l_ctx['avg_home']
            def_h = s_h_xg['xg_against'] if s_h_xg else l_ctx['avg_home_conceded']
            att_a = s_a_xg['xg_for'] if s_a_xg else l_ctx['avg_away']
            def_a = s_a_xg['xg_against'] if s_a_xg else l_ctx['avg_away_conceded']

            lh = l_ctx['avg_home'] * (att_h / l_ctx['avg_home']) * (def_a / l_ctx['avg_away_conceded'])
            la = l_ctx['avg_away'] * (att_a / l_ctx['avg_away']) * (def_h / l_ctx['avg_home_conceded'])

            # Matrice de Poisson
            matrix = np.zeros((7, 7))
            for x in range(7):
                for y in range(7):
                    matrix[x,y] = poisson.pmf(x, lh) * poisson.pmf(y, la)
            matrix /= matrix.sum()

            st.session_state.data = {
                'p_h': np.sum(np.tril(matrix, -1)), 'p_n': np.sum(np.diag(matrix)), 'p_a': np.sum(np.triu(matrix, 1)),
                'matrix': matrix, 't_h': t_h, 't_a': t_a, 'lh': lh, 'la': la
            }
            st.session_state.simulation_done = True

# --- AFFICHAGE DES RÉSULTATS ---
if st.session_state.simulation_done:
    d = st.session_state.data
    st.write("---")
    m1, m2, m3 = st.columns(3)
    m1.metric(d['t_h'], f"{d['p_h']*100:.1f}%")
    m2.metric("NUL", f"{d['p_n']*100:.1f}%")
    m3.metric(d['t_a'], f"{d['p_a']*100:.1f}%")

    st.subheader("🤖 PRONOSTIC")
    st.markdown("<div class='bet-card'>", unsafe_allow_html=True)
    c_h = st.number_input(f"Cote {d['t_h']}", value=2.0)
    ev = d['p_h'] * c_h
    
    verdict = "VALUE DÉTECTÉE" if ev > 1.05 else "PAS DE VALUE"
    st.markdown(f"<div class='verdict-text'>{verdict} | EV: {ev:.2f}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.subheader("SCORES LES PLUS PROBABLES")
    idx = np.unravel_index(np.argsort(d['matrix'].ravel())[-3:][::-1], d['matrix'].shape)
    for i in range(3):
        st.write(f"🎯 **{idx[0][i]} - {idx[1][i]}** ({d['matrix'][idx[0][i], idx[1][i]]*100:.1f}%)")

st.markdown("<br><center>iTrOz Predictor v2.5</center>", unsafe_allow_html=True)
