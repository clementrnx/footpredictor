import streamlit as st
import requests
import numpy as np
from scipy.stats import poisson
from datetime import datetime
import pandas as pd

# --- CONFIGURATION CLEMENTRNXX PREDICTOR V5.5 ---
st.set_page_config(page_title="Clementrnxx Predictor V5.5 - Final Edition", layout="wide")

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

API_KEY = st.secrets["MY_API_KEY"]
BASE_URL = "https://v3.football.api-sports.io/"
HEADERS = {'x-apisports-key': API_KEY}
SEASON = 2025

LEAGUES_DICT = {"La Liga": 140, "Premier League": 39, "Champions League": 2, "Ligue 1": 61, "Serie A": 135, "Bundesliga": 78}

# --- MOTEUR MATHÉMATIQUE Dixon-Coles ---
def dixon_coles_adj(x, y, lh, la, rho):
    if x == 0 and y == 0: return 1 - (lh * la * rho)
    if x == 0 and y == 1: return 1 + (lh * rho)
    if x == 1 and y == 0: return 1 + (la * rho)
    if x == 1 and y == 1: return 1 - rho
    return 1.0

def calculate_perfect_probs(lh, la):
    # rho = coefficient de corrélation (moyenne pro en Europe ~ -0.11)
    rho = -0.11
    matrix = np.zeros((10, 10))
    for x in range(10):
        for y in range(10):
            prob = poisson.pmf(x, lh) * poisson.pmf(y, la)
            adj = dixon_coles_adj(x, y, lh, la, rho)
            matrix[x, y] = max(0, prob * adj)
    
    matrix /= matrix.sum()
    p_h = np.sum(np.tril(matrix, -1))
    p_n = np.sum(np.diag(matrix))
    p_a = np.sum(np.triu(matrix, 1))
    
    return {
        "p_h": p_h, "p_n": p_n, "p_a": p_a,
        "p_1n": p_h + p_n, "p_n2": p_n + p_a, "p_12": p_h + p_a,
        "p_btts": np.sum(matrix[1:, 1:]), "matrix": matrix
    }

def get_optimized_lambda(team_id, league_id):
    f = get_api("fixtures", {"team": team_id, "season": SEASON, "last": 20})
    if not f: return 1.35
    
    # Calcul avec décroissance temporelle (Epsilon)
    goals = []
    for i, m in enumerate(f):
        g = (m['goals']['home'] if m['teams']['home']['id'] == team_id else m['goals']['away']) or 0
        # Plus le match est vieux (i élevé), moins il pèse
        weight = 0.96 ** i 
        goals.append(g * weight)
    
    return sum(goals) / sum([0.96**i for i in range(len(goals))])

@st.cache_data(ttl=3600)
def get_api(endpoint, params):
    try:
        r = requests.get(f"{BASE_URL}{endpoint}", headers=HEADERS, params=params, timeout=12)
        return r.json().get('response', [])
    except: return []

# --- UI ---
st.title("🏆 CLEMENTRNXX PREDICTOR V5.5")
st.subheader("FINAL EDITION - MATHEMATICAL PERFECTION")

tab1, tab2, tab3 = st.tabs(["🎯 ANALYSE 1VS1", "🚀 SCANNER DE TICKETS", "📊 STATS & TENDANCES"])

with tab1:
    l_name = st.selectbox("LIGUE", list(LEAGUES_DICT.keys()))
    teams_res = get_api("teams", {"league": LEAGUES_DICT[l_name], "season": SEASON})
    teams = {t['team']['name']: t['team']['id'] for t in teams_res}
    
    if teams:
        c1, c2 = st.columns(2)
        th, ta = c1.selectbox("DOMICILE", sorted(teams.keys())), c2.selectbox("EXTÉRIEUR", sorted(teams.keys()))
        if st.button("LANCER L'ANALYSE"):
            lh = get_optimized_lambda(teams[th], LEAGUES_DICT[l_name]) * 1.08 # Home Advantage
            la = get_optimized_lambda(teams[ta], LEAGUES_DICT[l_name]) * 0.92 # Away Factor
            st.session_state.final_cx = {"res": calculate_perfect_probs(lh, la), "th": th, "ta": ta}

    if 'final_cx' in st.session_state:
        r, th, ta = st.session_state.final_cx["res"], st.session_state.final_cx["th"], st.session_state.final_cx["ta"]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric(th, f"{r['p_h']*100:.2f}%")
        m2.metric("NUL", f"{r['p_n']*100:.2f}%")
        m3.metric(ta, f"{r['p_a']*100:.2f}%")
        m4.metric("BTTS", f"{r['p_btts']*100:.2f}%")

        st.subheader("🕵️ AUDIT TECHNIQUE")
        ac1, ac2 = st.columns(2)
        u_bet = ac1.selectbox("VOTRE CHOIX", [th, ta, "Nul", "1N", "N2", "12", "BTTS OUI"])
        u_odd = ac2.number_input("COTE", value=1.50)
        p_map = {th: r['p_h'], ta: r['p_a'], "Nul": r['p_n'], "1N": r['p_1n'], "N2": r['p_n2'], "12": r['p_12'], "BTTS OUI": r['p_btts']}
        ev = p_map[u_bet] * u_odd
        st.markdown(f"<div class='verdict-box'>EXPECTED VALUE (EV) : {ev:.4f}<br>VERDICT : {'MATHÉMATIQUEMENT RENTABLE' if ev > 1.00 else 'ESPÉRANCE NÉGATIVE'}</div>", unsafe_allow_html=True)

        st.subheader("🔢 TOP SCORES (Dixon-Coles Corrected)")
        idx = np.unravel_index(np.argsort(r['matrix'].ravel())[-5:][::-1], r['matrix'].shape)
        sc = st.columns(5)
        for i in range(5):
            sc[i].markdown(f"<div class='score-card'><b>{idx[0][i]} - {idx[1][i]}</b><br>{r['matrix'][idx[0][i],idx[1][i]]*100:.1f}%</div>", unsafe_allow_html=True)

with tab2:
    st.subheader("🚀 GÉNÉRATEUR DE TICKETS")
    risk_mode = st.select_slider("MODE DE RISQUE", options=["SAFE", "MID-SAFE", "MID", "MID-AGGRESSIF", "AGGRESSIF"], value="MID")
    risk_cfg = {"SAFE": {"p": 0.82, "ev": 1.02, "legs": 2}, "MID-SAFE": {"p": 0.74, "ev": 1.05, "legs": 3}, "MID": {"p": 0.64, "ev": 1.08, "legs": 4}, "MID-AGGRESSIF": {"p": 0.52, "ev": 1.12, "legs": 5}, "AGGRESSIF": {"p": 0.42, "ev": 1.15, "legs": 7}}[risk_mode]
    
    if st.button("SCANNER LE MARCHÉ"):
        opps = []
        for lid in LEAGUES_DICT.values():
            fixtures = get_api("fixtures", {"league": lid, "season": SEASON, "date": datetime.now().strftime('%Y-%m-%d')})
            for f in fixtures:
                lh, la = get_optimized_lambda(f['teams']['home']['id'], lid), get_optimized_lambda(f['teams']['away']['id'], lid)
                pr = calculate_perfect_probs(lh*1.08, la*0.92)
                tests = [("1", pr['p_h'], "Match Winner", "Home"), ("2", pr['p_a'], "Match Winner", "Away"), ("1N", pr['p_1n'], "Double Chance", "Home/Draw"), ("N2", pr['p_12'], "Double Chance", "Draw/Away"), ("BTTS", pr['p_btts'], "Both Teams Score", "Yes")]
                for lbl, p, m_n, m_v in tests:
                    if p >= risk_cfg['p']:
                        odds = get_api("odds", {"fixture": f['fixture']['id']})
                        if odds:
                            for btt in odds[0]['bookmakers'][0]['bets']:
                                if btt['name'] == m_n:
                                    for o in btt['values']:
                                        if o['value'] == m_v:
                                            ct = float(o['odd'])
                                            if (p * ct) >= risk_cfg['ev']:
                                                opps.append({"MATCH": f"{f['teams']['home']['name']}-{f['teams']['away']['name']}", "PARI": lbl, "COTE": ct, "EV": p*ct})
        
        final_t = sorted(opps, key=lambda x: x['EV'], reverse=True)[:risk_cfg['legs']]
        if final_t:
            st.markdown(f"<div class='verdict-box'>COTE TOTALE : @{np.prod([x['COTE'] for x in final_t]):.2f}</div>", unsafe_allow_html=True)
            st.table(final_t)

with tab3:
    st.subheader("📊 ANALYSE DE LIGUE")
    l_sel = st.selectbox("LIGUE POUR STATS", list(LEAGUES_DICT.keys()))
    standings = get_api("standings", {"league": LEAGUES_DICT[l_sel], "season": SEASON})
    if standings:
        df = pd.DataFrame([{"Equipe": t['team']['name'], "Pts": t['points'], "Forme": t['form'], "Buts+": t['all']['goals']['for']} for t in standings[0]['league']['standings'][0]])
        st.dataframe(df, use_container_width=True)

st.markdown(f"""<a href="https://github.com/clementrnx" class="github-link" target="_blank">GITHUB : github.com/clementrnx</a>""", unsafe_allow_html=True)
