import streamlit as st
import requests
import numpy as np
from scipy.stats import poisson
from datetime import datetime, timedelta

st.set_page_config(page_title="iTrOz Predictor", layout="wide")

# --- CSS AVEC EFFET AURA ET DISTORSION ---
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
        overflow: hidden;
    }

    .stApp::before {
        content: "";
        position: fixed;
        top: 0; left: 0; width: 100%; height: 100%;
        background: radial-gradient(circle at var(--mouse-x, 50%) var(--mouse-y, 50%), 
                    rgba(255, 215, 0, 0.15) 0%, 
                    rgba(0,0,0,0) 50%);
        pointer-events: none;
        z-index: 1;
    }

    .stApp > div:first-child { background-color: rgba(0, 0, 0, 0.85); position: relative; z-index: 2; }
    
    h1, h2, h3, p, span, label { color: #FFD700 !important; font-family: 'Monospace', sans-serif; letter-spacing: 2px; }

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

    div[data-baseweb="select"], div[data-baseweb="input"], .stNumberInput input, .stSelectbox div {
        background-color: rgba(255, 255, 255, 0.05) !important;
        backdrop-filter: blur(12px) !important;
        border: 0.5px solid rgba(255, 215, 0, 0.15) !important;
        border-radius: 10px !important;
        color: #FFD700 !important;
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
        border: 1px solid rgba(255, 215, 0, 0.05);
        margin-bottom: 40px;
    }

    .footer {
        text-align: center; padding: 50px 0 20px 0;
        color: rgba(255, 215, 0, 0.6); font-family: 'Monospace', sans-serif; font-size: 14px;
    }
    .footer a {
        color: #FFD700 !important; text-decoration: none; font-weight: bold;
        border: 1px solid rgba(255, 215, 0, 0.2); padding: 8px 15px; border-radius: 5px;
    }
    </style>

    <script>
    const doc = document.documentElement;
    document.addEventListener('mousemove', e => {
        doc.style.setProperty('--mouse-x', e.clientX + 'px');
        doc.style.setProperty('--mouse-y', e.clientY + 'px');
    });
    </script>
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
    except: 
        return []

@st.cache_data(ttl=3600)
def get_league_context(league_id, season):
    standings = get_api("standings", {"league": league_id, "season": season})
    if not standings or not standings[0].get('league', {}).get('standings'):
        return {'avg_home': 1.5, 'avg_away': 1.2, 'avg_home_conceded': 1.2, 'avg_away_conceded': 1.5, 'avg_total': 2.7}
    
    total_home_goals, total_away_goals, total_home_conceded, total_away_conceded, total_matches = 0, 0, 0, 0, 0
    for team in standings[0]['league']['standings'][0]:
        total_home_goals += team['home']['goals']['for']
        total_home_conceded += team['home']['goals']['against']
        total_away_goals += team['away']['goals']['for']
        total_away_conceded += team['away']['goals']['against']
        total_matches += team['home']['played']
    
    if total_matches == 0: return {'avg_home': 1.5, 'avg_away': 1.2, 'avg_total': 2.7}
    return {
        'avg_home': total_home_goals / total_matches,
        'avg_away': total_away_goals / total_matches,
        'avg_home_conceded': total_home_conceded / total_matches,
        'avg_away_conceded': total_away_conceded / total_matches,
        'avg_total': (total_home_goals + total_away_goals) / (total_matches * 2)
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
        
        if team_is_home:
            xg_f = float(f['teams']['home'].get('xg') or f['goals']['home'] or 0)
            xg_a = float(f['teams']['away'].get('xg') or f['goals']['away'] or 0)
        else:
            xg_f = float(f['teams']['away'].get('xg') or f['goals']['away'] or 0)
            xg_a = float(f['teams']['home'].get('xg') or f['goals']['home'] or 0)
            
        xg_f_w += xg_f * weight
        xg_a_w += xg_a * weight
        total_w += weight
        count += 1
    
    return {'xg_for': xg_f_w/total_w, 'xg_against': xg_a_w/total_w, 'matches_count': count} if total_w > 0 else None

@st.cache_data(ttl=1800)
def get_comprehensive_stats(team_id, league_id, season, use_global=False):
    return {
        'base': get_api("teams/statistics", {"league": league_id, "season": season, "team": team_id}),
        'xg_home': get_weighted_xg_stats(team_id, league_id, season, is_home=True, use_global=use_global),
        'xg_away': get_weighted_xg_stats(team_id, league_id, season, is_home=False, use_global=use_global)
    }

# --- MAIN INTERFACE ---
st.title("ITROZ PREDICTOR")

col_toggle, col_aggr, col_league = st.columns([1, 1, 2])
with col_toggle:
    use_global_stats = st.toggle("📊 MODE GLOBAL", value=False)
with col_aggr:
    aggressivity = st.select_slider("🎲 MODE", options=["PRUDENT", "ÉQUILIBRÉ", "JOUEUR", "RISQUÉ"], value="JOUEUR")

leagues = {"La Liga": 140, "Champions League": 2, "Premier League": 39, "Serie A": 135, "Bundesliga": 78, "Ligue 1": 61}
with col_league:
    l_name = st.selectbox("CHOISIR LA LIGUE", list(leagues.keys()))
l_id = leagues[l_name]

teams_res = get_api("teams", {"league": l_id, "season": SEASON})
teams = {t['team']['name']: t['team']['id'] for t in teams_res}

if teams:
    sorted_names = sorted(teams.keys())
    idx_h = next((i for i, n in enumerate(sorted_names) if "Barcelona" in n), 0)
    idx_a = next((i for i, n in enumerate(sorted_names) if "Real Madrid" in n), 1)

    c1, c2 = st.columns(2)
    t_h = c1.selectbox("DOMICILE", sorted_names, index=idx_h)
    t_a = c2.selectbox("EXTÉRIEUR", sorted_names, index=idx_a)

    if st.button("Lancer la prédiction"):
        id_h, id_a = teams[t_h], teams[t_a]
        with st.spinner("Analyse en cours..."):
            l_ctx = get_league_context(l_id, SEASON)
            s_h = get_comprehensive_stats(id_h, l_id, SEASON, use_global_stats)
            s_a = get_comprehensive_stats(id_a, l_id, SEASON, use_global_stats)
            
            # Calcul λ (Poisson)
            att_h = s_h['xg_home']['xg_for'] if s_h['xg_home'] else l_ctx['avg_home']
            def_h = s_h['xg_home']['xg_against'] if s_h['xg_home'] else l_ctx['avg_home_conceded']
            att_a = s_a['xg_away']['xg_for'] if s_a['xg_away'] else l_ctx['avg_away']
            def_a = s_a['xg_away']['xg_against'] if s_a['xg_away'] else l_ctx['avg_away_conceded']

            lh = l_ctx['avg_home'] * (att_h / l_ctx['avg_home']) * (def_a / l_ctx['avg_away_conceded'])
            la = l_ctx['avg_away'] * (att_a / l_ctx['avg_away']) * (def_h / l_ctx['avg_home_conceded'])

            # Matrice Dixon-Coles
            matrix = np.zeros((8, 8))
            for x in range(8):
                for y in range(8):
                    prob = poisson.pmf(x, lh) * poisson.pmf(y, la)
                    if x==0 and y==0: prob *= 0.87
                    elif (x==1 and y==0) or (x==0 and y==1): prob *= 1.06
                    matrix[x,y] = prob
            matrix /= matrix.sum()

            st.session_state.data = {
                'p_h': np.sum(np.tril(matrix, -1)), 'p_n': np.sum(np.diag(matrix)), 'p_a': np.sum(np.triu(matrix, 1)),
                'matrix': matrix, 't_h': t_h, 't_a': t_a, 'lh': lh, 'la': la, 'aggressivity': aggressivity
            }
            st.session_state.simulation_done = True

if st.session_state.simulation_done:
    d = st.session_state.data
    st.write("---")
    col1, col2, col3 = st.columns(3)
    col1.metric(d['t_h'], f"{d['p_h']*100:.1f}%")
    col2.metric("NUL", f"{d['p_n']*100:.1f}%")
    col3.metric(d['t_a'], f"{d['p_a']*100:.1f}%")

    st.subheader("🤖 MODE BET")
    st.markdown("<div class='bet-card'>", unsafe_allow_html=True)
    bankroll = st.number_input("CAPITAL TOTAL (€)", value=100.0)
    
    # Kelly & EV Logic
    c_h = st.number_input(f"COTE {d['t_h']}", value=2.0)
    ev = d['p_h'] * c_h
    if ev > 1.05:
        st.markdown(f"<div class='verdict-text'>IA RECOMMANDE : {d['t_h']} | EV: {ev:.2f}</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='verdict-text'>AUCUNE VALUE CLAIRE</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.subheader("SCORES PROBABLES")
    idx = np.unravel_index(np.argsort(d['matrix'].ravel())[-5:][::-1], d['matrix'].shape)
    score_cols = st.columns(5)
    for i in range(5):
        with score_cols[i]:
            st.write(f"**{idx[0][i]} - {idx[1][i]}**")
            st.write(f"{d['matrix'][idx[0][i], idx[1][i]]*100:.1f}%")

st.markdown("<div class='footer'>DÉVELOPPÉ PAR ITROZ | <a href='#'>GITHUB SOURCE</a></div>", unsafe_allow_html=True)
