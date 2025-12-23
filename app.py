import streamlit as st
import requests
import numpy as np
from scipy.stats import poisson
from datetime import datetime

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

# --- API Configuration ---
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

# --- Contexte ligue ---
@st.cache_data(ttl=3600)
def get_league_context(league_id, season):
    standings = get_api("standings", {"league": league_id, "season": season})
    if not standings or not standings[0].get('league', {}).get('standings'):
        return {'avg_home': 1.5, 'avg_away': 1.2, 'avg_total': 2.7}
    
    total_home_goals = total_away_goals = total_home_conceded = total_away_conceded = total_matches = 0
    
    for team in standings[0]['league']['standings'][0]:
        home_stats = team['home']
        away_stats = team['away']
        total_home_goals += home_stats['goals']['for']
        total_home_conceded += home_stats['goals']['against']
        total_away_goals += away_stats['goals']['for']
        total_away_conceded += away_stats['goals']['against']
        total_matches += home_stats['played']
    
    if total_matches == 0:
        return {'avg_home': 1.5, 'avg_away': 1.2, 'avg_total': 2.7}
    
    return {
        'avg_home': total_home_goals / total_matches,
        'avg_away': total_away_goals / total_matches,
        'avg_home_conceded': total_home_conceded / total_matches,
        'avg_away_conceded': total_away_conceded / total_matches,
        'avg_total': (total_home_goals + total_away_goals) / (total_matches * 2)
    }

# --- xG pondéré ---
@st.cache_data(ttl=1800)
def get_weighted_xg_stats(team_id, league_id, season, is_home=True, use_global=False):
    fixtures = get_api("fixtures", {"team": team_id, "season": season, "last": 15} if use_global else {"team": team_id, "league": league_id, "season": season, "last": 10})
    if not fixtures:
        return None
    xg_for_weighted = xg_against_weighted = goals_for_weighted = goals_against_weighted = total_weight = matches_count = 0
    fixtures_sorted = sorted(fixtures, key=lambda x: x['fixture']['date'], reverse=True)
    for idx, match in enumerate(fixtures_sorted):
        if match['fixture']['status']['short'] != 'FT':
            continue
        weight = 0.9 ** idx
        team_is_home = match['teams']['home']['id'] == team_id
        if is_home and not team_is_home: continue
        if not is_home and team_is_home: continue
        if team_is_home:
            xg_for = float(match['teams']['home'].get('xg') or match['goals']['home'] or 0)
            xg_against = float(match['teams']['away'].get('xg') or match['goals']['away'] or 0)
            goals_for = match['goals']['home'] or 0
            goals_against = match['goals']['away'] or 0
        else:
            xg_for = float(match['teams']['away'].get('xg') or match['goals']['away'] or 0)
            xg_against = float(match['teams']['home'].get('xg') or match['goals']['home'] or 0)
            goals_for = match['goals']['away'] or 0
            goals_against = match['goals']['home'] or 0
        xg_for_weighted += xg_for * weight
        xg_against_weighted += xg_against * weight
        goals_for_weighted += goals_for * weight
        goals_against_weighted += goals_against * weight
        total_weight += weight
        matches_count += 1
    if total_weight == 0 or matches_count == 0: return None
    return {
        'xg_for': xg_for_weighted / total_weight,
        'xg_against': xg_against_weighted / total_weight,
        'goals_for': goals_for_weighted / total_weight,
        'goals_against': goals_against_weighted / total_weight,
        'matches_count': matches_count
    }

@st.cache_data(ttl=1800)
def get_comprehensive_stats(team_id, league_id, season, use_global=False):
    base_stats = get_api("teams/statistics", {"league": league_id, "season": season, "team": team_id})
    xg_home = get_weighted_xg_stats(team_id, league_id, season, is_home=True, use_global=use_global)
    xg_away = get_weighted_xg_stats(team_id, league_id, season, is_home=False, use_global=use_global)
    return {'base': base_stats, 'xg_home': xg_home, 'xg_away': xg_away}

if 'simulation_done' not in st.session_state:
    st.session_state.simulation_done = False
    st.session_state.data = {}

st.title("ITROZ PREDICTOR")

# --- Mode global / ligue ---
col_toggle, col_league = st.columns([1, 3])
with col_toggle:
    use_global_stats = st.toggle("📊 MODE GLOBAL", value=False)
leagues = {"La Liga": 140, "Champions League": 2, "Premier League": 39, "Serie A": 135, "Bundesliga": 78, "Ligue 1": 61}
with col_league:
    l_name = st.selectbox("CHOISIR LA LIGUE", list(leagues.keys()))
l_id = leagues[l_name]

teams_res = get_api("teams", {"league": l_id, "season": SEASON})
teams = {t['team']['name']: t['team']['id'] for t in teams_res}

if teams:
    sorted_team_names = sorted(teams.keys())
    idx_barca = 0
    idx_real = 1
    for i, name in enumerate(sorted_team_names):
        if "Barcelona" in name: idx_barca = i
        if "Real Madrid" in name: idx_real = i
    c1, c2 = st.columns(2)
    t_h = c1.selectbox("DOMICILE", sorted_team_names, index=idx_barca)
    t_a = c2.selectbox("EXTÉRIEUR", sorted_team_names, index=idx_real)

    st.subheader("⚙️ Mode BET")
    bet_mode = st.selectbox("Choisir la stratégie", ["Conservateur", "Équilibré", "Agressif"])

    bankroll = st.number_input("CAPITAL TOTAL (€)", value=100.0)

    if st.button("Lancer la prédiction"):
        id_h, id_a = teams[t_h], teams[t_a]
        league_ctx = get_league_context(l_id, SEASON)
        stats_h = get_comprehensive_stats(id_h, l_id, SEASON, use_global=use_global_stats)
        stats_a = get_comprehensive_stats(id_a, l_id, SEASON, use_global=use_global_stats)

        s_h = stats_h['base']
        s_a = stats_a['base']

        # xG pondéré ou fallback
        att_h_home = stats_h['xg_home']['xg_for'] if stats_h['xg_home'] else league_ctx['avg_home']
        def_h_home = stats_h['xg_home']['xg_against'] if stats_h['xg_home'] else league_ctx['avg_home_conceded']
        att_a_away = stats_a['xg_away']['xg_for'] if stats_a['xg_away'] else league_ctx['avg_away']
        def_a_away = stats_a['xg_away']['xg_against'] if stats_a['xg_away'] else league_ctx['avg_away_conceded']

        attack_strength_h = att_h_home / league_ctx['avg_home']
        defense_weakness_a = def_a_away / league_ctx['avg_away_conceded']
        attack_strength_a = att_a_away / league_ctx['avg_away']
        defense_weakness_h = def_h_home / league_ctx['avg_home_conceded']

        lh = league_ctx['avg_home'] * attack_strength_h * defense_weakness_a
        la = league_ctx['avg_away'] * attack_strength_a * defense_weakness_h

        # Dixon-Coles
        tau_00, tau_10, tau_01, tau_11 = -0.13, 0.065, 0.065, 0.13
        max_goals = int(max(lh, la)*2.5)+3
        max_goals = min(max_goals, 10)
        matrix = np.zeros((max_goals, max_goals))
        for x in range(max_goals):
            for y in range(max_goals):
                prob = poisson.pmf(x, lh) * poisson.pmf(y, la)
                if x==0 and y==0: prob *= (1+tau_00*lh*la)
                elif x==1 and y==0: prob *= (1+tau_10*lh)
                elif x==0 and y==1: prob *= (1+tau_01*la)
                elif x==1 and y==1: prob *= (1+tau_11)
                matrix[x,y] = prob
        matrix = np.maximum(matrix, 0)
        matrix /= matrix.sum()

        p_h = np.sum(np.tril(matrix, -1))
        p_n = np.sum(np.diag(matrix))
        p_a = np.sum(np.triu(matrix, 1))

        st.session_state.data = {'matrix': matrix, 'p_h': p_h, 'p_n': p_n, 'p_a': p_a, 't_h': t_h, 't_a': t_a, 'lh': lh, 'la': la, 'bet_mode': bet_mode}

# --- Affichage BET et Audit ---
if st.session_state.simulation_done or 'data' in st.session_state:
    d = st.session_state.data
    st.write("---")
    
    # --- BET ---
    st.subheader("🤖 BET")
    best_o = max([
        {"n": d['t_h'], "p": d['p_h'], "c": 2.0},
        {"n": "NUL", "p": d['p_n'], "c": 3.0},
        {"n": d['t_a'], "p": d['p_a'], "c": 3.0}
    ], key=lambda x: x['p']*x['c'])
    
    if best_o['p']*best_o['c']>1.02:
        if d['bet_mode']=="Conservateur": pct=0.3
        elif d['bet_mode']=="Équilibré": pct=0.5
        else: pct=0.7
        stake = round(bankroll * pct,2)
        expected_value = round(stake * (best_o['p']*best_o['c'] - 1),2)
        st.markdown(f"<div class='verdict-text'>MEILLEUR PARI: {best_o['n']}<br>MISE: {stake} € ({pct*100:.0f}% du capital)<br>ESPÉRANCE: {expected_value} €</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='verdict-text'>AUCUN VALUE DÉTECTÉ</div>", unsafe_allow_html=True)

    # --- Audit du ticket ---
    st.subheader("📋 Audit du ticket")
    matrix = d['matrix']
    max_goals = matrix.shape[0]
    st.write("Probabilités de scores exacts:")
    for x in range(max_goals):
        for y in range(max_goals):
            if matrix[x,y]>0.01:
                st.write(f"{d['t_h']} {x} - {y} {d['t_a']}: {matrix[x,y]*100:.1f}%")
    
    st.write("🔹 Probabilité du résultat final:")
    st.write(f"- {d['t_h']} gagne: {d['p_h']*100:.1f}%")
    st.write(f"- Match nul: {d['p_n']*100:.1f}%")
    st.write(f"- {d['t_a']} gagne: {d['p_a']*100:.1f}%")

st.markdown("""
