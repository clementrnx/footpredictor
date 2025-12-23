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

# ----------------- CONFIGURATION API -----------------
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

# ----------------- FONCTIONS XG / DIXON -----------------
@st.cache_data(ttl=3600)
def get_league_context(league_id, season):
    standings = get_api("standings", {"league": league_id, "season": season})
    if not standings or not standings[0].get('league', {}).get('standings'):
        return {'avg_home': 1.5, 'avg_away': 1.2, 'avg_total': 2.7}
    total_home_goals = total_away_goals = total_home_conceded = total_away_conceded = total_matches = 0
    for team in standings[0]['league']['standings'][0]:
        home_stats = team['home']; away_stats = team['away']
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

@st.cache_data(ttl=1800)
def get_weighted_xg_stats(team_id, league_id, season, is_home=True, use_global=False):
    if use_global:
        fixtures = get_api("fixtures", {"team": team_id, "season": season, "last": 15})
    else:
        fixtures = get_api("fixtures", {"team": team_id, "league": league_id, "season": season, "last": 10})
    if not fixtures:
        return None
    xg_for_weighted = xg_against_weighted = goals_for_weighted = goals_against_weighted = total_weight = matches_count = 0
    fixtures_sorted = sorted(fixtures, key=lambda x: x['fixture']['date'], reverse=True)
    for idx, match in enumerate(fixtures_sorted):
        if match['fixture']['status']['short'] != 'FT': continue
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

# ----------------- SESSION -----------------
if 'simulation_done' not in st.session_state:
    st.session_state.simulation_done = False
    st.session_state.data = {}

st.title("ITROZ PREDICTOR")

# Choix du mode global
col_toggle, col_league = st.columns([1,3])
with col_toggle:
    use_global_stats = st.toggle("📊 MODE GLOBAL", value=False, help="Utilise stats toutes compétitions")

leagues = {"La Liga":140,"Champions League":2,"Premier League":39,"Serie A":135,"Bundesliga":78,"Ligue 1":61}
with col_league:
    l_name = st.selectbox("CHOISIR LA LIGUE", list(leagues.keys()))
l_id = leagues[l_name]

teams_res = get_api("teams", {"league": l_id, "season": SEASON})
teams = {t['team']['name']: t['team']['id'] for t in teams_res}

if teams:
    sorted_team_names = sorted(teams.keys())
    idx_barca = 0; idx_real = 1
    for i, name in enumerate(sorted_team_names):
        if "Barcelona" in name: idx_barca = i
        if "Real Madrid" in name: idx_real = i
    c1, c2 = st.columns(2)
    t_h = c1.selectbox("DOMICILE", sorted_team_names, index=idx_barca)
    t_a = c2.selectbox("EXTÉRIEUR", sorted_team_names, index=idx_real)

    if st.button("Lancer la prédiction"):
        id_h, id_a = teams[t_h], teams[t_a]
        league_ctx = get_league_context(l_id, SEASON)
        stats_h = get_comprehensive_stats(id_h, l_id, SEASON, use_global=use_global_stats)
        stats_a = get_comprehensive_stats(id_a, l_id, SEASON, use_global=use_global_stats)

        # Priorité xG pondéré
        if stats_h['xg_home'] and stats_h['xg_home']['matches_count']>=3:
            att_h_home = stats_h['xg_home']['xg_for']; def_h_home = stats_h['xg_home']['xg_against']
        else:
            att_h_home = float(stats_h['base'].get('goals',{}).get('for',{}).get('average',{}).get('home') or league_ctx['avg_home'])
            def_h_home = float(stats_h['base'].get('goals',{}).get('against',{}).get('average',{}).get('home') or league_ctx['avg_home_conceded'])
        if stats_a['xg_away'] and stats_a['xg_away']['matches_count']>=3:
            att_a_away = stats_a['xg_away']['xg_for']; def_a_away = stats_a['xg_away']['xg_against']
        else:
            att_a_away = float(stats_a['base'].get('goals',{}).get('for',{}).get('average',{}).get('away') or league_ctx['avg_away'])
            def_a_away = float(stats_a['base'].get('goals',{}).get('against',{}).get('average',{}).get('away') or league_ctx['avg_away_conceded'])

        # Forces relatives
        attack_strength_h = att_h_home / league_ctx['avg_home'] if league_ctx['avg_home']>0 else 1
        defense_weakness_a = def_a_away / league_ctx['avg_away_conceded'] if league_ctx['avg_away_conceded']>0 else 1
        attack_strength_a = att_a_away / league_ctx['avg_away'] if league_ctx['avg_away']>0 else 1
        defense_weakness_h = def_h_home / league_ctx['avg_home_conceded'] if league_ctx['avg_home_conceded']>0 else 1
        lh = league_ctx['avg_home'] * attack_strength_h * defense_weakness_a
        la = league_ctx['avg_away'] * attack_strength_a * defense_weakness_h

        # Dixon-Coles
        tau_00 = -0.13; tau_10=0.065; tau_01=0.065; tau_11=0.13
        max_goals = int(max(lh, la)*2.5)+3; max_goals=min(max_goals,10)
        matrix = np.zeros((max_goals,max_goals))
        for x in range(max_goals):
            for y in range(max_goals):
                prob = poisson.pmf(x,lh)*poisson.pmf(y,la)
                if x==0 and y==0: prob*=(1+tau_00*lh*la)
                elif x==1 and y==0: prob*=(1+tau_10*lh)
                elif x==0 and y==1: prob*=(1+tau_01*la)
                elif x==1 and y==1: prob*=(1+tau_11)
                matrix[x,y]=prob
        matrix=np.maximum(matrix,0)
        matrix/=matrix.sum()

        st.session_state.data = {
            'p_h': np.sum(np.tril(matrix,-1)),
            'p_n': np.sum(np.diag(matrix)),
            'p_a': np.sum(np.triu(matrix,1)),
            'matrix': matrix,
            't_h': t_h,
            't_a': t_a,
            'lh': lh, 'la': la,
            'league_avg': league_ctx['avg_total'],
            'using_xg_h': stats_h['xg_home'] is not None,
            'using_xg_a': stats_a['xg_away'] is not None,
            'xg_h_matches': stats_h['xg_home']['matches_count'] if stats_h['xg_home'] else 0,
            'xg_a_matches': stats_a['xg_away']['matches_count'] if stats_a['xg_away'] else 0,
            'mode': "Global" if use_global_stats else l_name
        }
        st.session_state.simulation_done=True

# ----------------- AFFICHAGE -----------------
if st.session_state.simulation_done:
    d = st.session_state.data
    st.write("---")
    # Métriques 3 colonnes
    m1,m2,m3 = st.columns(3)
    m1.metric(d['t_h'], f"{d['p_h']*100:.1f}%")
    m2.metric("NUL", f"{d['p_n']*100:.1f}%")
    m3.metric(d['t_a'], f"{d['p_a']*100:.1f}%")

    # ---------------- MODE BET -----------------
    st.subheader("🤖 MODE BET")
    st.markdown("<div class='bet-card'>", unsafe_allow_html=True)
    b1,b2,b3,b4 = st.columns(4)
    bankroll = b1.number_input("CAPITAL TOTAL (€)", value=100.0)
    c_h = b2.number_input(f"COTE {d['t_h']}", value=2.0)
    c_n = b3.number_input("COTE NUL", value=3.0)
    c_a = b4.number_input(f"COTE {d['t_a']}", value=3.0)
    opts = [
        {"name": d['t_h'], "p": d['p_h'], "c": c_h},
        {"name": "NUL", "p":
