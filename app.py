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
    .score-card { background: rgba(255, 255, 255, 0.07); border: 1px solid rgba(255, 215, 0, 0.4); padding: 15px; border-radius: 12px; text-align: center; }
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

RISK_LEVELS = {
    "SAFE": {"p": 0.78, "ev": 1.02, "kelly": 0.03, "legs": 2},
    "MID-SAFE": {"p": 0.72, "ev": 1.05, "kelly": 0.05, "legs": 3},
    "MID": {"p": 0.65, "ev": 1.08, "kelly": 0.08, "legs": 4},
    "MID-AGGRESSIF": {"p": 0.55, "ev": 1.12, "kelly": 0.12, "legs": 5},
    "AGGRESSIF": {"p": 0.45, "ev": 1.15, "kelly": 0.20, "legs": 7}
}

# --- FONCTIONS COEUR ---
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
    return {
        "p_h": p_h, "p_n": p_n, "p_a": p_a, 
        "p_1n": p_h+p_n, "p_n2": p_n+p_a, "p_12": p_h+p_a, 
        "p_btts": np.sum(matrix[1:, 1:]), "p_nobtts": 1.0 - np.sum(matrix[1:, 1:]),
        "lh": lh, "la": la
    }

# --- INTERFACE ---
st.title("⚽ CLEMENTRNXX PREDICTOR V5.5")

tab1, tab2, tab3 = st.tabs(["📊 ANALYSE 1VS1", "🔍 SCANNER DE TICKETS", "🏆 CLASSEMENTS"])

# --- MODULÉ 1 : ANALYSE 1VS1 RESTAURÉ ---
with tab1:
    l_name = st.selectbox("CHOISIR LA LIGUE", list(LEAGUES_DICT.keys()), key="l1")
    scope_1v1 = st.select_slider("PRÉCISION DATA", options=["LEAGUE ONLY", "OVER-ALL"], value="OVER-ALL")
    
    teams_res = get_api("teams", {"league": LEAGUES_DICT[l_name], "season": SEASON})
    teams = {t['team']['name']: t['team']['id'] for t in teams_res}
    
    if teams:
        c1, c2 = st.columns(2)
        th = c1.selectbox("DOMICILE", sorted(teams.keys()))
        ta = c2.selectbox("EXTÉRIEUR", sorted(teams.keys()))
        
        if st.button("LANCER L'ANALYSE"):
            with st.spinner("Calcul des probabilités Poisson..."):
                att_h, def_h = get_team_stats(teams[th], LEAGUES_DICT[l_name], scope_1v1=="OVER-ALL")
                att_a, def_a = get_team_stats(teams[ta], LEAGUES_DICT[l_name], scope_1v1=="OVER-ALL")
                lh, la = (att_h * def_a) ** 0.5 * 1.08, (att_a * def_h) ** 0.5 * 0.92
                st.session_state.res1v1 = {"r": calculate_perfect_probs(lh, la), "th": th, "ta": ta}

    if 'res1v1' in st.session_state:
        r, th, ta = st.session_state.res1v1["r"], st.session_state.res1v1["th"], st.session_state.res1v1["ta"]
        
        # Dashboard de Metrics
        cols = st.columns(5)
        cols[0].metric(th, f"{r['p_h']:.1%}")
        cols[1].metric("NUL", f"{r['p_n']:.1%}")
        cols[2].metric(ta, f"{r['p_a']:.1%}")
        cols[3].metric("BTTS OUI", f"{r['p_btts']:.1%}")
        cols[4].metric("BTTS NON", f"{r['p_nobtts']:.1%}")

        # Module de calcul de mise
        st.subheader("💰 CALCULATEUR DE VALUE")
        mc1, mc2 = st.columns([2, 1])
        with mc2:
            bk = st.number_input("SOLDE (€)", value=100.0)
            rk = st.selectbox("RISQUE STRATÉGIQUE", list(RISK_LEVELS.keys()), index=2)
        with mc1:
            i1, i2, i3 = st.columns(3)
            c_h = i1.number_input(f"Cote {th}", value=1.0)
            c_hn = i2.number_input(f"Cote {th}/N", value=1.0)
            c_n2 = i3.number_input(f"Cote N/{ta}", value=1.0)
            
        cfg = RISK_LEVELS[rk]
        potentials = [
            (f"Victoire {th}", c_h, r['p_h']), 
            (f"Double Chance {th}/N", c_hn, r['p_1n']), 
            (f"Double Chance N/{ta}", c_n2, r['p_n2'])
        ]
        
        for name, cote, prob in potentials:
            if cote > 1.0 and (prob * cote) >= cfg['ev'] and prob >= cfg['p']:
                st.success(f"🔥 VALUE DÉTECTÉE : {name} | Cote @{cote} | Mise conseillée : {bk * cfg['kelly']:.2f}€")

# --- MODULE 2 : SCANNER DE TICKETS RESTAURÉ ---
with tab2:
    st.subheader("🔎 SCANNER AUTOMATIQUE")
    sc1, sc2, sc3 = st.columns(3)
    l_scan = sc1.selectbox("LIGUES À SCANNER", ["TOUTES"] + list(LEAGUES_DICT.keys()))
    days = sc2.number_input("SCANNER SUR X JOURS", 1, 7, 1)
    risk_s = sc3.selectbox("PROFIL DE TICKET", list(RISK_LEVELS.keys()), index=1)
    
    selected_markets = st.multiselect("MARCHÉS AUTORISÉS", ["ISSUE SIMPLE", "DOUBLE CHANCE", "BTTS"], default=["ISSUE SIMPLE", "DOUBLE CHANCE"])
    
    if st.button("GÉNÉRER LE TICKET IDÉAL"):
        cfg_s = RISK_LEVELS[risk_s]
        opps = []
        lids = LEAGUES_DICT.values() if l_scan == "TOUTES" else [LEAGUES_DICT[l_scan]]
        
        progress = st.progress(0)
        for i in range(days):
            date_str = (datetime.now() + timedelta(days=i)).strftime('%Y-%m-%d')
            for lid in lids:
                fixtures = get_api("fixtures", {"league": lid, "season": SEASON, "date": date_str})
                for f in fixtures:
                    if f['fixture']['status']['short'] != "NS": continue
                    # Stats & Probas
                    att_h, def_h = get_team_stats(f['teams']['home']['id'], lid, True)
                    att_a, def_a = get_team_stats(f['teams']['away']['id'], lid, True)
                    lh, la = (att_h * def_a) ** 0.5 * 1.08, (att_a * def_h) ** 0.5 * 0.92
                    p = calculate_perfect_probs(lh, la)
                    
                    # Construction des tests
                    tests = []
                    if "ISSUE SIMPLE" in selected_markets:
                        tests += [(f['teams']['home']['name'], p['p_h'], "Match Winner", "Home", 1.1)] # Boost léger pour issue simple
                        tests += [(f['teams']['away']['name'], p['p_a'], "Match Winner", "Away", 1.1)]
                    if "DOUBLE CHANCE" in selected_markets:
                        tests += [(f"{f['teams']['home']['name']}/N", p['p_1n'], "Double Chance", "Home/Draw", 1.0)]
                        tests += [(f"N/{f['teams']['away']['name']}", p['p_n2'], "Double Chance", "Draw/Away", 1.0)]
                    if "BTTS" in selected_markets:
                        tests += [("BTTS OUI", p['p_btts'], "Both Teams Score", "Yes", 1.0)]

                    for lbl, prob, m_n, m_v, boost in tests:
                        if prob >= cfg_s['p']:
                            odds_data = get_api("odds", {"fixture": f['fixture']['id']})
                            if odds_data:
                                for bk_info in odds_data[0].get('bookmakers', []):
                                    for bet in bk_info.get('bets', []):
                                        if bet['name'] == m_n:
                                            for val in bet['values']:
                                                if val['value'] == m_v:
                                                    cote = float(val['odd'])
                                                    if (prob * cote) >= cfg_s['ev']:
                                                        opps.append({
                                                            "MATCH": f"{f['teams']['home']['name']} vs {f['teams']['away']['name']}",
                                                            "PARI": lbl, "COTE": cote, "PROBA": f"{prob:.1%}",
                                                            "SCORE": prob * cote * boost
                                                        })
                                    break
            progress.progress((i + 1) / days)
            
        final_list = sorted(opps, key=lambda x: x['SCORE'], reverse=True)[:cfg_s['legs']]
        if final_list:
            total_odd = np.prod([x['COTE'] for x in final_list])
            st.markdown(f"<div class='verdict-box'>🚀 TICKET GÉNÉRÉ | COTE TOTALE : @{total_odd:.2f}</div>", unsafe_allow_html=True)
            st.table(final_list)
        else:
            st.warning("Aucune opportunité trouvée respectant tes critères de risque.")

# --- MODULE 3 : CLASSEMENTS ---
with tab3:
    l_stand = st.selectbox("LIGUE STANDINGS", list(LEAGUES_DICT.keys()))
    data = get_api("standings", {"league": LEAGUES_DICT[l_stand], "season": SEASON})
    if data:
        s = data[0]['league']['standings'][0]
        df = pd.DataFrame([{"Pos": x['rank'], "Equipe": x['team']['name'], "Pts": x['points'], "Forme": x['form']} for x in s])
        st.dataframe(df, use_container_width=True)

st.markdown("<br><hr><center><a href='https://github.com/clementrnx' class='github-link'>Developed by CLEMENTRNXX</a></center>", unsafe_allow_html=True)
