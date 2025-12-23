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

# --- CONFIG API ---
API_KEY = st.secrets["MY_API_KEY"]
BASE_URL = "https://v3.football.api-sports.io/"
HEADERS = {'x-apisports-key': API_KEY}
SEASON = 2025

LEAGUES_DICT = {"La Liga": 140, "Premier League": 39, "Champions League": 2, "Ligue 1": 61, "Serie A": 135, "Bundesliga": 78}

# --- MOTEUR MATHÉMATIQUE ---
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
    matrix /= matrix.sum()
    p_h, p_n, p_a = np.sum(np.tril(matrix, -1)), np.sum(np.diag(matrix)), np.sum(np.triu(matrix, 1))
    p_btts = np.sum(matrix[1:, 1:])
    return {
        "p_h": p_h, "p_n": p_n, "p_a": p_a, 
        "p_1n": p_h+p_n, "p_n2": p_n+p_a, "p_12": p_h+p_a, 
        "p_btts": p_btts, "p_nobtts": 1.0 - p_btts, "matrix": matrix
    }

def get_optimized_lambda(team_id, league_id, scope_overall):
    params = {"team": team_id, "season": SEASON, "last": 15}
    if not scope_overall: params["league"] = league_id
    f = get_api("fixtures", params)
    if not f: return 1.35
    goals = [(m['goals']['home'] if m['teams']['home']['id'] == team_id else m['goals']['away']) or 0 for m in f]
    weights = [0.95 ** i for i in range(len(goals))]
    return sum(g * w for g, w in zip(goals, weights)) / sum(weights)

@st.cache_data(ttl=3600)
def get_api(endpoint, params):
    try:
        r = requests.get(f"{BASE_URL}{endpoint}", headers=HEADERS, params=params, timeout=12)
        return r.json().get('response', [])
    except: return []

# --- INTERFACE ---
st.title("🏆 CLEMENTRNXX PREDICTOR V5.5")
st.subheader("FINAL EDITION")

tab1, tab2, tab3 = st.tabs(["🎯 ANALYSE 1VS1", "🚀 SCANNER DE TICKETS", "📊 STATS"])

with tab1:
    l_name = st.selectbox("LIGUE DU MATCH", list(LEAGUES_DICT.keys()))
    scope_1v1 = st.select_slider("MODE DATA SOURCE", options=["LEAGUE ONLY", "OVER-ALL"], value="OVER-ALL")
    
    teams_res = get_api("teams", {"league": LEAGUES_DICT[l_name], "season": SEASON})
    teams = {t['team']['name']: t['team']['id'] for t in teams_res}
    
    if teams:
        c1, c2 = st.columns(2)
        th, ta = c1.selectbox("DOMICILE", sorted(teams.keys())), c2.selectbox("EXTÉRIEUR", sorted(teams.keys()))
        if st.button("LANCER L'ANALYSE"):
            lh = get_optimized_lambda(teams[th], LEAGUES_DICT[l_name], scope_1v1=="OVER-ALL") * 1.08
            la = get_optimized_lambda(teams[ta], LEAGUES_DICT[l_name], scope_1v1=="OVER-ALL") * 0.92
            st.session_state.v5_final = {"res": calculate_perfect_probs(lh, la), "th": th, "ta": ta}

    if 'v5_final' in st.session_state:
        r, th, ta = st.session_state.v5_final["res"], st.session_state.v5_final["th"], st.session_state.v5_final["ta"]
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric(th, f"{r['p_h']*100:.1f}%")
        m2.metric("MATCH NUL", f"{r['p_n']*100:.1f}%")
        m3.metric(ta, f"{r['p_a']*100:.1f}%")
        m4.metric("BTTS OUI", f"{r['p_btts']*100:.1f}%")
        m5.metric("BTTS NON", f"{r['p_nobtts']*100:.1f}%")

        st.subheader("🕵️ AUDIT DU PARI")
        ac1, ac2 = st.columns(2)
        u_bet = ac1.selectbox("VOTRE SÉLECTION", [th, ta, "Match Nul", f"{th} ou Nul", f"Nul ou {ta}", f"{th} ou {ta}", "Les deux équipes marquent", "Une ou aucune équipe marque"])
        u_odd = ac2.number_input("COTE DU BOOKMAKER", value=1.50, step=0.01)
        
        p_map = {th: r['p_h'], ta: r['p_a'], "Match Nul": r['p_n'], f"{th} ou Nul": r['p_1n'], f"Nul ou {ta}": r['p_n2'], f"{th} ou {ta}": r['p_12'], "Les deux équipes marquent": r['p_btts'], "Une ou aucune équipe marque": r['p_nobtts']}
        ev = p_map[u_bet] * u_odd
        st.markdown(f"<div class='verdict-box'>EXPECTED VALUE : {ev:.4f}<br>STATUT : {'✅ VALIDE' if ev > 1.05 else '❌ EV NÉGATIVE'}</div>", unsafe_allow_html=True)

        st.subheader("💰 MODULE BET (COMPARATIF)")
        b1, b2, b3 = st.columns(3)
        c_1 = b1.number_input(f"Cote {th}", value=1.0)
        c_n = b2.number_input("Cote Nul", value=1.0)
        c_2 = b3.number_input(f"Cote {ta}", value=1.0)
        b4, b5, b6, b7, b8 = st.columns(5)
        c_1n = b4.number_input(f"{th}/N", value=1.0)
        c_n2 = b5.number_input(f"N/{ta}", value=1.0)
        c_12 = b6.number_input(f"{th}/{ta}", value=1.0)
        c_btts = b7.number_input("BTTS OUI", value=1.0)
        c_nobtts = b8.number_input("BTTS NON", value=1.0)

        cap = st.number_input("BANKROLL (€)", value=100.0)
        k = max(0, (((u_odd-1) * p_map[u_bet]) - (1 - p_map[u_bet])) / (u_odd-1)) if u_odd > 1 else 0
        st.success(f"MISE OPTIMISÉE : **{(cap * k * 0.2):.2f} €**")

        st.subheader("🔢 TOP SCORES")
        idx = np.unravel_index(np.argsort(r['matrix'].ravel())[-5:][::-1], r['matrix'].shape)
        sc = st.columns(5)
        for i in range(5): sc[i].markdown(f"<div class='score-card'><b>{idx[0][i]} - {idx[1][i]}</b><br>{r['matrix'][idx[0][i],idx[1][i]]*100:.1f}%</div>", unsafe_allow_html=True)

with tab2:
    st.subheader("🚀 GÉNÉRATEUR DE TICKETS")
    gc1, gc2, gc3 = st.columns(3)
    l_scan = gc1.selectbox("CHAMPIONNAT", ["TOUTES LES LEAGUES"] + list(LEAGUES_DICT.keys()))
    d_scan = gc2.date_input("DATE DU SCAN", datetime.now())
    scope_scan = gc3.select_slider("DATA SCAN", options=["LEAGUE ONLY", "OVER-ALL"], value="OVER-ALL")
    
    selected_markets = st.multiselect("MARCHÉS À ANALYSER", ["ISSUE SIMPLE", "DOUBLE CHANCE", "BTTS (OUI/NON)"], default=["ISSUE SIMPLE", "DOUBLE CHANCE"])
    risk_mode = st.select_slider("MODE DE RISQUE", options=["SAFE", "MID-SAFE", "MID", "MID-AGGRESSIF", "AGGRESSIF"], value="MID")
    risk_cfg = {"SAFE": {"p": 0.82, "ev": 1.02, "legs": 2}, "MID-SAFE": {"p": 0.74, "ev": 1.05, "legs": 3}, "MID": {"p": 0.64, "ev": 1.08, "legs": 4}, "MID-AGGRESSIF": {"p": 0.52, "ev": 1.12, "legs": 5}, "AGGRESSIF": {"p": 0.42, "ev": 1.15, "legs": 7}}[risk_mode]
    
    if st.button("GÉNÉRER TICKET"):
        lids = LEAGUES_DICT.values() if l_scan == "TOUTES LES LEAGUES" else [LEAGUES_DICT[l_scan]]
        opps = []
        for lid in lids:
            fixtures = get_api("fixtures", {"league": lid, "season": SEASON, "date": d_scan.strftime('%Y-%m-%d')})
            for f in fixtures:
                lh = get_optimized_lambda(f['teams']['home']['id'], lid, scope_scan=="OVER-ALL") * 1.08
                la = get_optimized_lambda(f['teams']['away']['id'], lid, scope_scan=="OVER-ALL") * 0.92
                pr = calculate_perfect_probs(lh, la)
                h_name, a_name = f['teams']['home']['name'], f['teams']['away']['name']
                
                tests = []
                if "ISSUE SIMPLE" in selected_markets:
                    tests += [(h_name, pr['p_h'], "Match Winner", "Home"), (a_name, pr['p_a'], "Match Winner", "Away")]
                if "DOUBLE CHANCE" in selected_markets:
                    tests += [(f"{h_name}/N", pr['p_1n'], "Double Chance", "Home/Draw"), (f"N/{a_name}", pr['p_n2'], "Double Chance", "Draw/Away")]
                if "BTTS (OUI/NON)" in selected_markets:
                    tests += [("BTTS OUI", pr['p_btts'], "Both Teams Score", "Yes"), ("BTTS NON", pr['p_nobtts'], "Both Teams Score", "No")]

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
                                                opps.append({"MATCH": f"{h_name}-{a_name}", "PARI": lbl, "COTE": ct, "PROBA": f"{p*100:.1f}%", "VALUE": p*ct})
        
        final_ticket = sorted(opps, key=lambda x: x['VALUE'], reverse=True)[:risk_cfg['legs']]
        if final_ticket:
            st.markdown(f"<div class='verdict-box'>TICKET GÉNÉRÉ | COTE : @{np.prod([x['COTE'] for x in final_ticket]):.2f}</div>", unsafe_allow_html=True)
            st.table(final_ticket)
        else: st.error("Aucune opportunité détectée pour cette configuration.")

with tab3:
    st.subheader("📊 CLASSEMENTS")
    l_sel = st.selectbox("LIGUE", list(LEAGUES_DICT.keys()))
    standings = get_api("standings", {"league": LEAGUES_DICT[l_sel], "season": SEASON})
    if standings:
        df = pd.DataFrame([{"Equipe": t['team']['name'], "Pts": t['points'], "Forme": t['form'], "Buts+": t['all']['goals']['for']} for t in standings[0]['league']['standings'][0]])
        st.dataframe(df, use_container_width=True)

st.markdown("""<a href="https://github.com/clementrnx" class="github-link" target="_blank" style="color:#FFD700;">GITHUB : github.com/clementrnx</a>""", unsafe_allow_html=True)
