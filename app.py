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

# SEUILS FLEXIBLES : On accepte tout ce qui a une espérance de gain positive (EV > 1)
RISK_LEVELS = {
    "ULTRA-SAFE": {"p": 0.75, "ev": 1.02, "kelly": 0.03, "legs": 2}, 
    "SAFE": {"p": 0.68, "ev": 1.04, "kelly": 0.05, "legs": 3},
    "MID": {"p": 0.60, "ev": 1.08, "kelly": 0.08, "legs": 4},
    "AGGRESSIF": {"p": 0.45, "ev": 1.15, "kelly": 0.15, "legs": 6}
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
st.subheader("STRATÉGIE : PURE VALUE SCANNER")

tab1, tab2 = st.tabs([" ANALYSE 1VS1", " SCANNER DE TICKETS"])

with tab1:
    l_name = st.selectbox("LIGUE DU MATCH", list(LEAGUES_DICT.keys()))
    scope_1v1 = st.select_slider("MODE DATA SOURCE", options=["LEAGUE ONLY", "OVER-ALL"], value="OVER-ALL")
    teams_res = get_api("teams", {"league": LEAGUES_DICT[l_name], "season": SEASON})
    teams = {t['team']['name']: t['team']['id'] for t in teams_res}
    
    if teams:
        c1, c2 = st.columns(2)
        team_h = c1.selectbox("DOMICILE", sorted(teams.keys()), key="th_1v1")
        team_a = c2.selectbox("EXTÉRIEUR", sorted(teams.keys()), key="ta_1v1")
        if st.button("LANCER L'ANALYSE"):
            att_h, def_h = get_team_stats(teams[team_h], LEAGUES_DICT[l_name], scope_1v1=="OVER-ALL")
            att_a, def_a = get_team_stats(teams[team_a], LEAGUES_DICT[l_name], scope_1v1=="OVER-ALL")
            lh, la = (att_h * def_a) ** 0.5 * 1.08, (att_a * def_h) ** 0.5 * 0.92
            st.session_state.v5_final = {"res": calculate_perfect_probs(lh, la), "th": team_h, "ta": team_a}

with tab2:
    st.subheader(" GÉNÉRATEUR DE TICKETS SANS FILTRE")
    gc1, gc2, gc3, gc4 = st.columns(4)
    l_scan = gc1.selectbox("CHAMPIONNAT", ["TOUTES LES LEAGUES"] + list(LEAGUES_DICT.keys()), key="l_scan")
    d_range = gc2.date_input("PÉRIODE", [datetime.now(), datetime.now()], key="d_scan_range")
    bank_scan = gc3.number_input("FOND (€)", value=100.0, key="b_scan_input")
    risk_mode = st.selectbox("RISQUE", list(RISK_LEVELS.keys()), index=1, key="risk_scan")
    risk_cfg = RISK_LEVELS[risk_mode]
    
    if st.button("GÉNÉRER LE TICKET", key="btn_gen"):
        if isinstance(d_range, (list, tuple)) and len(d_range) == 2:
            start_date, end_date = d_range
            date_list = pd.date_range(start=start_date, end=end_date).tolist()
        else:
            st.error("Sélectionnez les deux dates")
            st.stop()

        lids = LEAGUES_DICT.values() if l_scan == "TOUTES LES LEAGUES" else [LEAGUES_DICT[l_scan]]
        opps = []
        progress_bar = st.progress(0)
        
        for idx_d, current_date in enumerate(date_list):
            date_str = current_date.strftime('%Y-%m-%d')
            for lid in lids:
                fixtures = get_api("fixtures", {"league": lid, "season": SEASON, "date": date_str})
                for f in fixtures:
                    if f['fixture']['status']['short'] != "NS": continue
                    att_h, def_h = get_team_stats(f['teams']['home']['id'], lid, True)
                    att_a, def_a = get_team_stats(f['teams']['away']['id'], lid, True)
                    lh, la = (att_h * def_a) ** 0.5 * 1.08, (att_a * def_h) ** 0.5 * 0.92
                    pr, h_n, a_n = calculate_perfect_probs(lh, la), f['teams']['home']['name'], f['teams']['away']['name']
                    
                    # On scanne les issues simples et doubles chances
                    tests = [(h_n, pr['p_h'], "Match Winner", "Home"), 
                             (a_n, pr['p_a'], "Match Winner", "Away"),
                             (f"{h_n}/N", pr['p_1n'], "Double Chance", "Home/Draw"), 
                             (f"N/{a_n}", pr['p_n2'], "Double Chance", "Draw/Away")]

                    for lbl, p, m_n, m_v in tests:
                        if p >= risk_cfg['p']:
                            odds_data = get_api("odds", {"fixture": f['fixture']['id']})
                            if odds_data:
                                for btt in odds_data[0].get('bookmakers', []):
                                    for bet in btt.get('bets', []):
                                        if bet['name'] == m_n:
                                            for val in bet['values']:
                                                if val['value'] == m_v:
                                                    ct = float(val['odd'])
                                                    # AUCUN FILTRE DE COTE MINIMALE
                                                    # Seul le critère EV (p * cote) est conservé pour la rentabilité
                                                    if (p * ct) >= risk_cfg['ev']:
                                                        opps.append({
                                                            "MATCH": f"{h_n}-{a_n}", "PARI": lbl, 
                                                            "COTE": ct, "PROBA": f"{p*100:.1f}%", 
                                                            "VALUE": p * ct
                                                        })
                                    break
            progress_bar.progress((idx_d + 1) / len(date_list))
        
        # Le ticket idéal prend les meilleures opportunités (Value la plus haute)
        final_ticket = sorted(opps, key=lambda x: x['VALUE'], reverse=True)[:risk_cfg['legs']]
        if final_ticket:
            total_odd = np.prod([x['COTE'] for x in final_ticket])
            st.markdown(f"<div class='verdict-box'>COTE TOTALE : @{total_odd:.2f} | MISE SUGGÉRÉE : {bank_scan * risk_cfg['kelly']:.2f}€</div>", unsafe_allow_html=True)
            st.table(final_ticket)
        else: st.error("Aucune opportunité correspondant aux réglages.")

st.markdown("""<a href="https://github.com/clementrnx" class="github-link" target="_blank" style="color:#FFD700;">GITHUB : github.com/clementrnx</a>""", unsafe_allow_html=True)
