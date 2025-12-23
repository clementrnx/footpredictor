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
    .stApp > div:first-child { background-color: rgba(0, 0, 0, 0.94); }
    h1, h2, h3, p, span, label { color: #FFD700 !important; font-family: 'Monospace', sans-serif; }
    div.stButton > button {
        background: rgba(255, 215, 0, 0.1) !important; backdrop-filter: blur(10px);
        border: 2px solid #FFD700 !important; color: #FFD700 !important;
        border-radius: 15px !important; font-weight: 900; transition: 0.4s; width: 100%;
    }
    div.stButton > button:hover { background: #FFD700 !important; color: black !important; box-shadow: 0 0 30px rgba(255, 215, 0, 0.6); }
    .metric-container { background: rgba(0,0,0,0.6); padding: 15px; border-radius: 15px; border: 1px solid #FFD700; text-align: center; }
    .github-link { display: block; text-align: center; color: #FFD700 !important; font-weight: bold; font-size: 1.2rem; text-decoration: none; margin-top: 40px; padding: 20px; border-top: 1px solid rgba(255, 215, 0, 0.2); }
    </style>
""", unsafe_allow_html=True)

API_KEY = st.secrets["MY_API_KEY"]
BASE_URL = "https://v3.football.api-sports.io/"
HEADERS = {'x-apisports-key': API_KEY}
SEASON = 2025

LEAGUES_DICT = {"La Liga": 140, "Premier League": 39, "Champions League": 2, "Ligue 1": 61, "Serie A": 135, "Bundesliga": 78}

# --- MOTEUR MATHÉMATIQUE Dixon-Coles ---
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
    
    # Calcul des probabilités de base
    p_h = np.sum(np.tril(matrix, -1))
    p_n = np.sum(np.diag(matrix))
    p_a = np.sum(np.triu(matrix, 1))
    
    return {
        "1": p_h, "N": p_n, "2": p_a,
        "1N": p_h + p_n, "N2": p_n + p_a, "12": p_h + p_a,
        "BTTS_YES": np.sum(matrix[1:, 1:]),
        "BTTS_NO": 1 - np.sum(matrix[1:, 1:]),
        "O2.5": np.sum([matrix[x,y] for x in range(10) for y in range(10) if x+y > 2.5]),
        "U2.5": np.sum([matrix[x,y] for x in range(10) for y in range(10) if x+y < 2.5]),
        "matrix": matrix
    }

def get_optimized_lambda(team_id, league_id, scope_overall):
    params = {"team": team_id, "season": SEASON, "last": 15}
    if not scope_overall: params["league"] = league_id
    f = get_api("fixtures", params)
    if not f: return 1.35
    goals = [(m['goals']['home'] if m['teams']['home']['id'] == team_id else m['goals']['away']) or 0 for i, m in enumerate(f)]
    weights = [0.95 ** i for i in range(len(goals))]
    return sum(g * w for g, w in zip(goals, weights)) / sum(weights)

@st.cache_data(ttl=3600)
def get_api(endpoint, params):
    try:
        r = requests.get(f"{BASE_URL}{endpoint}", headers=HEADERS, params=params, timeout=12)
        return r.json().get('response', [])
    except: return []

# --- UI ---
st.title("🏆 CLEMENTRNXX PREDICTOR V5.5")
st.subheader("FINAL EDITION - PRO BETTING ENGINE")

tab1, tab2, tab3 = st.tabs(["🎯 ANALYSE 1VS1", "🚀 SCANNER DE TICKETS", "📊 STATS"])

with tab1:
    l_name = st.selectbox("LIGUE", list(LEAGUES_DICT.keys()))
    scope_1v1 = st.select_slider("MODE DATA", options=["LEAGUE ONLY", "OVER-ALL"], value="OVER-ALL")
    
    teams_res = get_api("teams", {"league": LEAGUES_DICT[l_name], "season": SEASON})
    teams = {t['team']['name']: t['team']['id'] for t in teams_res}
    
    if teams:
        c1, c2 = st.columns(2)
        th, ta = c1.selectbox("DOMICILE", sorted(teams.keys())), c2.selectbox("EXTÉRIEUR", sorted(teams.keys()))
        if st.button("LANCER L'ANALYSE EXPERTE"):
            lh = get_optimized_lambda(teams[th], LEAGUES_DICT[l_name], (scope_1v1 == "OVER-ALL")) * 1.08
            la = get_optimized_lambda(teams[ta], LEAGUES_DICT[l_name], (scope_1v1 == "OVER-ALL")) * 0.92
            st.session_state.v5_pro = {"res": calculate_perfect_probs(lh, la), "th": th, "ta": ta}

    if 'v5_pro' in st.session_state:
        r, th, ta = st.session_state.v5_pro["res"], st.session_state.v5_pro["th"], st.session_state.v5_pro["ta"]
        
        # DISPLAY PROBAS
        cols = st.columns(4)
        cols[0].metric(th, f"{r['1']*100:.1f}%")
        cols[1].metric("NUL", f"{r['N']*100:.1f}%")
        cols[2].metric(ta, f"{r['2']*100:.1f}%")
        cols[3].metric("BTTS", f"{r['BTTS_YES']*100:.1f}%")

        st.markdown("---")
        # AUDIT & BET (INTEGRÉS POUR PERFORMANCE)
        col_audit, col_bet = st.columns(2)
        
        with col_audit:
            st.subheader("🕵️ AUDIT UNIVERSEL")
            market = st.selectbox("MARCHÉ CIBLE", ["Victoire 1", "Nul", "Victoire 2", "1N", "N2", "12", "BTTS OUI", "BTTS NON", "Plus 2.5", "Moins 2.5", "Score Exact"])
            
            p_final = 0
            if market == "Score Exact":
                s_h = st.number_input("Buts Domicile", 0, 9, 1)
                s_a = st.number_input("Buts Extérieur", 0, 9, 1)
                p_final = r['matrix'][s_h, s_a]
            else:
                mapping = {"Victoire 1": "1", "Nul": "N", "Victoire 2": "2", "1N": "1N", "N2": "N2", "12": "12", "BTTS OUI": "BTTS_YES", "BTTS NON": "BTTS_NO", "Plus 2.5": "O2.5", "Moins 2.5": "U2.5"}
                p_final = r[mapping[market]]
            
            u_odd = st.number_input("COTE DU BOOKMAKER", value=2.0, step=0.01)
            fair_odd = 1/p_final if p_final > 0 else 0
            ev = p_final * u_odd
            
            st.markdown(f"""<div class='metric-container'>
                COTE IA : <b>{fair_odd:.2f}</b><br>
                EXPECTED VALUE : <b>{ev:.4f}</b><br>
                STATUS : {'✅ VALUE DETECTED' if ev > 1.05 else '❌ NO VALUE'}
            </div>""", unsafe_allow_html=True)

        with col_bet:
            st.subheader("💰 MODE BET PRO")
            bankroll = st.number_input("VOTRE BANKROLL (€)", value=100.0)
            # Kelly Criterion Fractionné (Prudence 0.2)
            b = u_odd - 1
            kelly = max(0, ((b * p_final) - (1 - p_final)) / b) if b > 0 else 0
            mise = bankroll * kelly * 0.2
            
            st.markdown(f"""<div class='metric-container' style='background: rgba(255,215,0,0.1);'>
                MISE CONSEILLÉE : <br><span style='font-size: 24px;'>{mise:.2f} €</span><br>
                ({(kelly*0.2*100):.1f}% de la bankroll)
            </div>""", unsafe_allow_html=True)

with tab2:
    st.subheader("🚀 SCANNER HAUT RENDEMENT")
    l_scan = st.selectbox("CHAMPIONNAT", ["TOUTES LES LEAGUES"] + list(LEAGUES_DICT.keys()))
    risk_mode = st.select_slider("MODE DE RISQUE", options=["SAFE", "MID-SAFE", "MID", "MID-AGGRESSIF", "AGGRESSIF"], value="MID")
    risk_cfg = {"SAFE": {"p": 0.82, "ev": 1.02, "legs": 2}, "MID-SAFE": {"p": 0.74, "ev": 1.05, "legs": 3}, "MID": {"p": 0.64, "ev": 1.08, "legs": 4}, "MID-AGGRESSIF": {"p": 0.52, "ev": 1.12, "legs": 5}, "AGGRESSIF": {"p": 0.42, "ev": 1.15, "legs": 7}}[risk_mode]
    
    if st.button("LANCER LE SCAN"):
        lids = LEAGUES_DICT.values() if l_scan == "TOUTES LES LEAGUES" else [LEAGUES_DICT[l_scan]]
        opps = []
        for lid in lids:
            fixtures = get_api("fixtures", {"league": lid, "season": SEASON, "date": datetime.now().strftime('%Y-%m-%d')})
            for f in fixtures:
                lh = get_optimized_lambda(f['teams']['home']['id'], lid, True) * 1.08
                la = get_optimized_lambda(f['teams']['away']['id'], lid, True) * 0.92
                pr = calculate_perfect_probs(lh, la)
                
                # Check all major markets
                tests = [("1", pr['1'], "Match Winner", "Home"), ("2", pr['2'], "Match Winner", "Away"), ("1N", pr['1N'], "Double Chance", "Home/Draw"), ("N2", pr['N2'], "Double Chance", "Draw/Away"), ("BTTS", pr['BTTS_YES'], "Both Teams Score", "Yes")]
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
                                                opps.append({"MATCH": f"{f['teams']['home']['name']}-{f['teams']['away']['name']}", "PARI": lbl, "COTE IA": round(1/p,2), "COTE BOOK": ct, "EV": round(p*ct,3)})
        
        final_ticket = sorted(opps, key=lambda x: x['EV'], reverse=True)[:risk_cfg['legs']]
        if final_ticket:
            st.table(final_ticket)
            st.success(f"Cote Totale Estimée : @{np.prod([x['COTE BOOK'] for x in final_ticket]):.2f}")

with tab3:
    st.subheader("📊 CLASSEMENTS")
    l_sel = st.selectbox("CHOISIR LIGUE", list(LEAGUES_DICT.keys()))
    standings = get_api("standings", {"league": LEAGUES_DICT[l_sel], "season": SEASON})
    if standings:
        df = pd.DataFrame([{"Equipe": t['team']['name'], "Pts": t['points'], "Forme": t['form'], "Buts+": t['all']['goals']['for']} for t in standings[0]['league']['standings'][0]])
        st.dataframe(df, use_container_width=True)

st.markdown(f"""<a href="https://github.com/clementrnx" class="github-link" target="_blank">GITHUB : github.com/clementrnx</a>""", unsafe_allow_html=True)
