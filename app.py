import streamlit as st
import requests
import numpy as np
from scipy.stats import poisson
from datetime import datetime, timedelta
import pandas as pd
import time

# --- CONFIGURATION CLEMENTRNXX PREDICTOR V5.5 ---
st.set_page_config(page_title="Clementrnxx Predictor V5.5", layout="wide")

st.markdown("""
    <style>
    .stApp { background-image: url("https://media.giphy.com/media/VZrfUvQjXaGEQy1RSn/giphy.gif"); background-size: cover; background-attachment: fixed; }
    .stApp > div:first-child { background-color: rgba(0, 0, 0, 0.93); }
    h1, h2, h3, p, span, label { color: #FFD700 !important; font-family: 'Monospace', sans-serif; }
    div.stButton > button {
        background: rgba(255, 215, 0, 0.1) !important; backdrop-filter: blur(10px);
        border: 2px solid #FFD700 !important; color: #FFD700 !important;
        border-radius: 15px !important; font-weight: 900; transition: 0.4s; width: 100%;
    }
    div.stButton > button:hover { background: #FFD700 !important; color: black !important; box-shadow: 0 0 30px rgba(255, 215, 0, 0.6); }
    .verdict-box { border: 2px solid #FFD700; padding: 25px; text-align: center; border-radius: 20px; background: rgba(0,0,0,0.8); margin: 20px 0; }
    .github-link { display: block; text-align: center; color: #FFD700 !important; font-weight: bold; font-size: 1.2rem; text-decoration: none; margin-top: 40px; padding: 20px; border-top: 1px solid rgba(255, 215, 0, 0.2); }
    </style>
""", unsafe_allow_html=True)

# --- CONFIG API & RISK ---
API_KEY = st.secrets["MY_API_KEY"]
DISCORD_WEBHOOK = "https://discord.com/api/webhooks/1453026279275106355/gbYAwBRntm1FCoqoBTz5lj1SCe2ijyeHHYoe4CFYwpzOw2DO-ozcCsgkK_53HhB-kFGE"
BASE_URL = "https://v3.football.api-sports.io/"
HEADERS = {'x-apisports-key': API_KEY}
SEASON = 2025

LEAGUES_DICT = {"La Liga": 140, "Premier League": 39, "Champions League": 2, "Ligue 1": 61, "Serie A": 135, "Bundesliga": 78}

# RÉINTÉGRATION DE TOUS LES NIVEAUX DE RISQUE
RISK_LEVELS = {
    "SAFE": {"p": 0.75, "ev": 1.02, "kelly": 0.03, "legs": 2},
    "MID-SAFE": {"p": 0.68, "ev": 1.05, "kelly": 0.05, "legs": 3},
    "MID": {"p": 0.60, "ev": 1.08, "kelly": 0.08, "legs": 4},
    "MID-AGGRESSIF": {"p": 0.52, "ev": 1.12, "kelly": 0.12, "legs": 5},
    "AGGRESSIF": {"p": 0.42, "ev": 1.15, "kelly": 0.20, "legs": 7}
}

# --- FONCTIONS ---
@st.cache_data(ttl=3600)
def get_api(endpoint, params):
    try:
        r = requests.get(f"{BASE_URL}{endpoint}", headers=HEADERS, params=params, timeout=12)
        if r.status_code == 429: time.sleep(1); return get_api(endpoint, params)
        return r.json().get('response', [])
    except: return []

def get_team_stats(team_id, league_id, scope_overall):
    params = {"team": team_id, "season": SEASON, "last": 15}
    if not scope_overall: params["league"] = league_id
    f = get_api("fixtures", params)
    if not f: return 1.2, 1.2
    scored, conceded = [], []
    for m in f:
        if m['goals']['home'] is not None:
            is_home = m['teams']['home']['id'] == team_id
            scored.append(m['goals']['home'] if is_home else m['goals']['away'])
            conceded.append(m['goals']['away'] if is_home else m['goals']['home'])
    if not scored: return 1.2, 1.2
    weights = [0.95 ** i for i in range(len(scored))]
    sum_w = sum(weights)
    return sum(s * w for s, w in zip(scored, weights)) / sum_w, sum(c * w for c, w in zip(conceded, weights)) / sum_w

def calculate_perfect_probs(lh, la):
    rho = -0.11
    matrix = np.zeros((10, 10))
    for x in range(10):
        for y in range(10):
            prob = poisson.pmf(x, lh) * poisson.pmf(y, la)
            adj = 1.0
            if x == 0 and y == 0: adj = 1 - (lh * la * rho)
            elif x == 0 and y == 1: adj = 1 + (lh * rho)
            elif x == 1 and y == 0: adj = 1 + (la * rho)
            elif x == 1 and y == 1: adj = 1 - rho
            matrix[x, y] = max(0, prob * adj)
    if matrix.sum() > 0: matrix /= matrix.sum()
    p_h, p_n, p_a = np.sum(np.tril(matrix, -1)), np.sum(np.diag(matrix)), np.sum(np.triu(matrix, 1))
    return {"p_h": p_h, "p_n": p_n, "p_a": p_a, "p_1n": p_h+p_n, "p_n2": p_n+p_a, "p_12": p_h+p_a, "p_btts": np.sum(matrix[1:, 1:]), "p_nobtts": 1.0 - np.sum(matrix[1:, 1:]), "matrix": matrix}

# --- UI ---
st.title(" CLEMENTRNXX PREDICTOR V5.5")

tab1, tab2 = st.tabs([" ANALYSE 1VS1", " SCANNER DE TICKETS"])

with tab1:
    l_name = st.selectbox("LIGUE", list(LEAGUES_DICT.keys()))
    teams_res = get_api("teams", {"league": LEAGUES_DICT[l_name], "season": SEASON})
    teams = {t['team']['name']: t['team']['id'] for t in teams_res}
    if teams:
        c1, c2 = st.columns(2)
        th = c1.selectbox("DOMICILE", sorted(teams.keys()))
        ta = c2.selectbox("EXTÉRIEUR", sorted(teams.keys()))
        if st.button("ANALYSER"):
            att_h, def_h = get_team_stats(teams[th], LEAGUES_DICT[l_name], True)
            att_a, def_a = get_team_stats(teams[ta], LEAGUES_DICT[l_name], True)
            lh, la = (att_h * def_a) ** 0.5 * 1.08, (att_a * def_h) ** 0.5 * 0.92
            r = calculate_perfect_probs(lh, la)
            st.write(f"Probabilité {th}: {r['p_h']:.1%}, Nul: {r['p_n']:.1%}, {ta}: {r['p_a']:.1%}")

with tab2:
    st.subheader("GÉNÉRATEUR MULTI-RISQUES")
    gc1, gc2, gc3 = st.columns(3)
    d_range = gc1.date_input("PÉRIODE", [datetime.now(), datetime.now()])
    bank = gc2.number_input("BANKROLL (€)", value=100.0)
    risk_mode = gc3.selectbox("MODE DE RISQUE", list(RISK_LEVELS.keys()))
    risk_cfg = RISK_LEVELS[risk_mode]

    if st.button("GÉNÉRER TICKET"):
        opps = []
        date_list = pd.date_range(start=d_range[0], end=d_range[1]).tolist()
        
        for current_date in date_list:
            date_str = current_date.strftime('%Y-%m-%d')
            for lid in LEAGUES_DICT.values():
                fixtures = get_api("fixtures", {"league": lid, "season": SEASON, "date": date_str})
                for f in fixtures:
                    if f['fixture']['status']['short'] != "NS": continue
                    att_h, def_h = get_team_stats(f['teams']['home']['id'], lid, True)
                    att_a, def_a = get_team_stats(f['teams']['away']['id'], lid, True)
                    lh, la = (att_h * def_a) ** 0.5 * 1.08, (att_a * def_h) ** 0.5 * 0.92
                    pr, h_n, a_n = calculate_perfect_probs(lh, la), f['teams']['home']['name'], f['teams']['away']['name']
                    
                    # On liste les opportunités
                    tests = [
                        (h_n, pr['p_h'], "Match Winner", "Home", "SIMPLE"),
                        (a_n, pr['p_a'], "Match Winner", "Away", "SIMPLE"),
                        (f"{h_n}/N", pr['p_1n'], "Double Chance", "Home/Draw", "DOUBLE"),
                        (f"N/{a_n}", pr['p_n2'], "Double Chance", "Draw/Away", "DOUBLE")
                    ]

                    for lbl, p, m_n, m_v, m_type in tests:
                        if p >= risk_cfg['p']:
                            odds_data = get_api("odds", {"fixture": f['fixture']['id']})
                            if odds_data:
                                for btt in odds_data[0].get('bookmakers', []):
                                    for bet in btt.get('bets', []):
                                        if bet['name'] == m_n:
                                            for val in bet['values']:
                                                if val['value'] == m_v:
                                                    ct = float(val['odd'])
                                                    # L'EV doit être positive. On booste légèrement le score des Issues Simples
                                                    # pour qu'elles ne soient pas systématiquement évincées par les doubles issues.
                                                    score = p * ct
                                                    if m_type == "SIMPLE": score *= 1.1 
                                                    
                                                    if (p * ct) >= risk_cfg['ev']:
                                                        opps.append({
                                                            "MATCH": f"{h_n}-{a_n}", "PARI": lbl, 
                                                            "COTE": ct, "PROBA": f"{p*100:.1f}%", 
                                                            "VALUE": score
                                                        })
                                    break
        
        final_ticket = sorted(opps, key=lambda x: x['VALUE'], reverse=True)[:risk_cfg['legs']]
        if final_ticket:
            total_odd = np.prod([x['COTE'] for x in final_ticket])
            st.markdown(f"<div class='verdict-box'>COTE TOTALE : @{total_odd:.2f} | MISE : {bank * risk_cfg['kelly']:.2f}€</div>", unsafe_allow_html=True)
            st.table(final_ticket)
        else: st.error("Rien trouvé.")
