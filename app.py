import streamlit as st
import requests
import numpy as np
from scipy.stats import poisson
from datetime import datetime, timedelta

# --- CONFIGURATION ET STYLE ORIGINAL RESTAURÉ ---
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

.stApp > div:first-child {
    background-color: rgba(0, 0, 0, 0.85);
    position: relative;
    z-index: 2;
}

h1, h2, h3, p, span, label {
    color: #FFD700 !important;
    font-family: 'Monospace', sans-serif;
    letter-spacing: 2px;
}

/* Boutons */
div.stButton > button {
    background: rgba(255, 215, 0, 0.03) !important;
    backdrop-filter: blur(25px) !important;
    border: 1px solid rgba(255, 215, 0, 0.2) !important;
    color: #FFD700 !important;
    border-radius: 15px !important;
    height: 60px !important;
    width: 100% !important;
    font-weight: 200 !important;
    text-transform: uppercase !important;
    letter-spacing: 6px !important;
    transition: 0.6s all ease-in-out;
}

div.stButton > button:hover {
    background: rgba(255, 215, 0, 0.1) !important;
    border: 1px solid rgba(255, 215, 0, 0.6) !important;
    letter-spacing: 8px !important;
    box-shadow: 0 0 40px rgba(255, 215, 0, 0.15);
}

/* Inputs glass */
div[data-baseweb="select"],
div[data-baseweb="input"],
.stNumberInput input,
.stSelectbox div {
    background-color: rgba(255, 255, 255, 0.05) !important;
    backdrop-filter: blur(12px) !important;
    border: 0.5px solid rgba(255, 215, 0, 0.15) !important;
    border-radius: 10px !important;
    color: #FFD700 !important;
}

/* ❌ SUPPRESSION DES TRAITS / OUTLINES */
div[data-baseweb="input"] > div,
div[data-baseweb="select"] > div,
.stNumberInput input,
.stTextInput input {
    border-bottom: none !important;
    box-shadow: none !important;
}

input:focus,
textarea:focus,
select:focus {
    outline: none !important;
    box-shadow: none !important;
}

/* ❌ Supprime les séparateurs --- */
hr {
    display: none !important;
}

/* ❌ Neutralise BaseWeb */
[data-baseweb] * {
    box-shadow: none !important;
}

.verdict-text {
    font-size: 26px;
    font-weight: 900;
    text-align: center;
    padding: 30px;
    letter-spacing: 4px;
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
</style>
""", unsafe_allow_html=True)

# --- API CONFIG ---
API_KEY = st.secrets["MY_API_KEY"]
BASE_URL = "https://v3.football.api-sports.io/"
HEADERS = {"x-apisports-key": API_KEY}
SEASON = 2025

@st.cache_data(ttl=3600)
def get_api(endpoint, params):
    try:
        r = requests.get(f"{BASE_URL}{endpoint}", headers=HEADERS, params=params, timeout=12)
        return r.json().get("response", [])
    except:
        return []

@st.cache_data(ttl=3600)
def get_league_context(league_id, season):
    return {
        "avg_home": 1.5,
        "avg_away": 1.2,
        "avg_home_conceded": 1.2,
        "avg_away_conceded": 1.5,
        "avg_total": 2.7,
    }

@st.cache_data(ttl=1800)
def get_weighted_xg_stats(team_id, league_id, season, is_home=True, use_global=False):
    params = (
        {"team": team_id, "season": season, "last": 15}
        if use_global
        else {"team": team_id, "league": league_id, "season": season, "last": 10}
    )
    fixtures = get_api("fixtures", params)
    if not fixtures:
        return None

    total_w, xg_f, xg_a = 0, 0, 0
    for idx, match in enumerate(fixtures):
        if match["fixture"]["status"]["short"] != "FT":
            continue
        weight = 0.9 ** idx
        home = match["teams"]["home"]["id"] == team_id
        if (is_home and not home) or (not is_home and home):
            continue
        xg_f += (
            float(match["goals"]["home" if home else "away"] or 0) * weight
        )
        xg_a += (
            float(match["goals"]["away" if home else "home"] or 0) * weight
        )
        total_w += weight

    return (
        {"xg_for": xg_f / total_w, "xg_against": xg_a / total_w}
        if total_w > 0
        else None
    )

# --- APP ---
st.title("ITROZ PREDICTOR")

col_toggle, col_league = st.columns([1, 3])
with col_toggle:
    use_global_stats = st.toggle("📊 MODE GLOBAL", value=False)

leagues = {
    "La Liga": 140,
    "Champions League": 2,
    "Premier League": 39,
    "Serie A": 135,
    "Bundesliga": 78,
    "Ligue 1": 61,
}

with col_league:
    l_name = st.selectbox("CHOISIR LA LIGUE", list(leagues.keys()))

teams_res = get_api("teams", {"league": leagues[l_name], "season": SEASON})
teams = {t["team"]["name"]: t["team"]["id"] for t in teams_res}

if teams:
    c1, c2 = st.columns(2)
    t_h = c1.selectbox("DOMICILE", sorted(teams.keys()), index=0)
    t_a = c2.selectbox("EXTÉRIEUR", sorted(teams.keys()), index=1)

    if st.button("LANCER LA PRÉDICTION"):
        ctx = get_league_context(leagues[l_name], SEASON)
        s_h = get_weighted_xg_stats(teams[t_h], leagues[l_name], SEASON, True, use_global_stats)
        s_a = get_weighted_xg_stats(teams[t_a], leagues[l_name], SEASON, False, use_global_stats)

        if s_h and s_a:
            lh = ctx["avg_home"] * (s_h["xg_for"] / ctx["avg_home"]) * (
                s_a["xg_against"] / ctx["avg_home_conceded"]
            )
            la = ctx["avg_away"] * (s_a["xg_for"] / ctx["avg_away"]) * (
                s_h["xg_against"] / ctx["avg_away_conceded"]
            )

            matrix = np.zeros((8, 8))
            for x in range(8):
                for y in range(8):
                    prob = poisson.pmf(x, lh) * poisson.pmf(y, la)
                    if x == 0 and y == 0:
                        prob *= 0.87
                    matrix[x, y] = prob

            matrix /= matrix.sum()
            st.session_state.data = {
                "p_h": np.sum(np.tril(matrix, -1)),
                "p_n": np.sum(np.diag(matrix)),
                "p_a": np.sum(np.triu(matrix, 1)),
                "matrix": matrix,
                "t_h": t_h,
                "t_a": t_a,
            }
            st.session_state.simulation_done = True

if st.session_state.get("simulation_done"):
    d = st.session_state.data

    m1, m2, m3 = st.columns(3)
    m1.metric(d["t_h"], f"{d['p_h']*100:.1f}%")
    m2.metric("NUL", f"{d['p_n']*100:.1f}%")
    m3.metric(d["t_a"], f"{d['p_a']*100:.1f}%")

st.markdown("""
<div style="text-align:center; padding:30px;">
    <a href="https://github.com/clementrnx" target="_blank" style="text-decoration:none;">
        <button style="
            background: rgba(255, 215, 0, 0.03);
            backdrop-filter: blur(25px);
            border: 1px solid rgba(255, 215, 0, 0.25);
            color: #FFD700;
            border-radius: 15px;
            height: 60px;
            padding: 0 40px;
            font-weight: 200;
            text-transform: uppercase;
            letter-spacing: 6px;
            cursor: pointer;
            transition: 0.6s all ease-in-out;
        ">
            🔗 GITHUB · CLEMENTRNX
        </button>
    </a>
</div>
<div style="text-align:center; padding:20px; opacity:0.6;">
    DÉVELOPPÉ PAR ITROZ
</div>
""", unsafe_allow_html=True)
