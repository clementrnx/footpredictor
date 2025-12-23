import streamlit as st
import requests
import numpy as np
from scipy.stats import poisson

# ================== CONFIG ==================
st.set_page_config(page_title="iTrOz Predictor", layout="wide")

API_KEY = st.secrets["MY_API_KEY"]
BASE_URL = "https://v3.football.api-sports.io/"
HEADERS = {'x-apisports-key': API_KEY}
SEASON = 2025
MAX_STAKE = 0.70

# ================== API ==================
@st.cache_data(ttl=3600)
def get_api(endpoint, params):
    try:
        r = requests.get(f"{BASE_URL}{endpoint}", headers=HEADERS, params=params, timeout=10)
        return r.json().get('response', [])
    except:
        return []

@st.cache_data(ttl=3600)
def get_league_context(league_id, season):
    standings = get_api("standings", {"league": league_id, "season": season})
    if not standings:
        return {'avg_home': 1.5, 'avg_away': 1.2, 'avg_home_conceded': 1.2, 'avg_away_conceded': 1.5}

    th, ta, th_c, ta_c, m = 0, 0, 0, 0, 0
    for t in standings[0]['league']['standings'][0]:
        th += t['home']['goals']['for']
        th_c += t['home']['goals']['against']
        ta += t['away']['goals']['for']
        ta_c += t['away']['goals']['against']
        m += t['home']['played']

    return {
        'avg_home': th / m,
        'avg_away': ta / m,
        'avg_home_conceded': th_c / m,
        'avg_away_conceded': ta_c / m
    }

# ================== BET LOGIC ==================
def kelly_fraction(p, c):
    b = c - 1
    return max(((b * p) - (1 - p)) / b, 0) if b > 0 else 0

def model_confidence(d):
    score = 0
    score += 1 if d['using_xg_h'] else 0
    score += 1 if d['using_xg_a'] else 0
    score += 1 if abs(d['lh'] - d['la']) > 0.6 else 0
    return score  # 0 à 3

def mode_conservative(signals, bankroll):
    v = [s for s in signals if s['ev'] >= 1.05 and s['confidence'] >= 2]
    if not v: return None
    b = max(v, key=lambda x: x['ev'])
    return b, bankroll * min(b['kelly'] * 0.35, MAX_STAKE)

def mode_balanced(signals, bankroll):
    v = [s for s in signals if s['ev'] >= 1.02 and s['confidence'] >= 1]
    if not v: return None
    b = max(v, key=lambda x: x['ev'])
    return b, bankroll * min(b['kelly'] * 0.70, MAX_STAKE)

def mode_aggressive(signals, bankroll):
    v = [s for s in signals if s['ev'] >= 1.00]
    if not v: return None
    b = max(v, key=lambda x: x['ev'])
    return b, bankroll * min(b['kelly'], MAX_STAKE)

def mode_value(signals, bankroll):
    v = [s for s in signals if s['ev'] > 1.00]
    if not v: return None
    b = max(v, key=lambda x: x['ev'])
    return b, bankroll * min(0.10, MAX_STAKE)

# ================== UI ==================
st.title("ITROZ PREDICTOR")

leagues = {"La Liga": 140, "Premier League": 39, "Serie A": 135}
league = st.selectbox("LIGUE", leagues.keys())
league_id = leagues[league]

teams_res = get_api("teams", {"league": league_id, "season": SEASON})
teams = {t['team']['name']: t['team']['id'] for t in teams_res}

c1, c2 = st.columns(2)
home = c1.selectbox("DOMICILE", teams.keys())
away = c2.selectbox("EXTÉRIEUR", teams.keys())

if st.button("Lancer la prédiction"):
    ctx = get_league_context(league_id, SEASON)

    lh = ctx['avg_home']
    la = ctx['avg_away']

    max_g = 10
    matrix = np.zeros((max_g, max_g))
    for i in range(max_g):
        for j in range(max_g):
            matrix[i, j] = poisson.pmf(i, lh) * poisson.pmf(j, la)
    matrix /= matrix.sum()

    p_h = np.sum(np.tril(matrix, -1))
    p_n = np.sum(np.diag(matrix))
    p_a = np.sum(np.triu(matrix, 1))

    st.metric(home, f"{p_h*100:.1f}%")
    st.metric("NUL", f"{p_n*100:.1f}%")
    st.metric(away, f"{p_a*100:.1f}%")

    st.subheader("MODE BET")

    bankroll = st.number_input("CAPITAL (€)", value=100.0)
    c_h = st.number_input(f"COTE {home}", value=2.0)
    c_n = st.number_input("COTE NUL", value=3.0)
    c_a = st.number_input(f"COTE {away}", value=3.0)

    bet_mode = st.selectbox(
        "MODE DE BET",
        ["CONSERVATIVE", "BALANCED", "AGGRESSIVE", "PURE VALUE"]
    )

    opts = [
        {"n": home, "p": p_h, "c": c_h},
        {"n": "NUL", "p": p_n, "c": c_n},
        {"n": away, "p": p_a, "c": c_a}
    ]

    d = {
        "lh": lh, "la": la,
        "using_xg_h": True,
        "using_xg_a": True
    }

    conf = model_confidence(d)

    signals = []
    for o in opts:
        signals.append({
            "name": o['n'],
            "p": o['p'],
            "c": o['c'],
            "ev": o['p'] * o['c'],
            "kelly": kelly_fraction(o['p'], o['c']),
            "confidence": conf
        })

    if bet_mode == "CONSERVATIVE":
        res = mode_conservative(signals, bankroll)
    elif bet_mode == "BALANCED":
        res = mode_balanced(signals, bankroll)
    elif bet_mode == "AGGRESSIVE":
        res = mode_aggressive(signals, bankroll)
    else:
        res = mode_value(signals, bankroll)

    if res:
        o, stake = res
        st.markdown(
            f"### ✅ {bet_mode}\n"
            f"**Pari :** {o['name']}\n\n"
            f"**Mise :** {stake:.2f} €\n\n"
            f"EV: {o['ev']:.2f} | Kelly: {o['kelly']:.2f} | Conf: {o['confidence']}/3"
        )
    else:
        st.warning("Aucun pari recommandé dans ce mode.")
