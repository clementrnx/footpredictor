import streamlit as st
import requests
import numpy as np
from scipy.stats import poisson
from datetime import datetime

# --- CONFIGURATION ET STYLE ---
st.set_page_config(page_title="iTrOz Predictor V4.8 - No-Match Edition", layout="wide")

st.markdown("""
    <style>
    .stApp {
        background-image: url("https://media.giphy.com/media/VZrfUvQjXaGEQy1RSn/giphy.gif");
        background-size: cover;
        background-attachment: fixed;
    }
    .stApp > div:first-child { background-color: rgba(0, 0, 0, 0.94); }
    h1, h2, h3, p, span, label { color: #FFD700 !important; font-family: 'Monospace', sans-serif; }
    
    .verdict-box {
        border: 2px solid #FFD700; padding: 20px; text-align: center;
        border-radius: 15px; background: rgba(255, 215, 0, 0.05); margin: 15px 0;
    }
    .bet-card {
        background: rgba(255, 255, 255, 0.03); padding: 20px; border-radius: 12px;
        border: 1px solid rgba(255, 215, 0, 0.15); margin-bottom: 20px;
    }
    div.stButton > button {
        background: rgba(255, 215, 0, 0.1) !important; border: 1px solid #FFD700 !important;
        color: #FFD700 !important; font-weight: bold; width: 100%;
    }
    </style>
""", unsafe_allow_html=True)

# --- CONFIG API ---
API_KEY = st.secrets["MY_API_KEY"]
BASE_URL = "https://v3.football.api-sports.io/"
HEADERS = {'x-apisports-key': API_KEY}
SEASON = 2025

# Configuration No-Match optimisée (Cotes dès 1.12)
ALGO_MODES = {
    "ULTRA SAFE (NO-MATCH)": {"min_p": 0.75, "min_cote": 1.12, "max_cote": 1.45, "min_ev": 1.02, "max_legs": 3},
    "SEREIN (MID)": {"min_p": 0.60, "min_cote": 1.30, "max_cote": 1.90, "min_ev": 1.05, "max_legs": 4},
    "VALUE (CHASSEUR)": {"min_p": 0.45, "min_cote": 1.60, "max_cote": 3.00, "min_ev": 1.10, "max_legs": 5}
}

LEAGUES_DICT = {
    "🌍 TOUS LES CHAMPIONNATS": "ALL",
    "🇪🇸 La Liga": 140, "🇬🇧 Premier League": 39, "🇪🇺 Champions League": 2, 
    "🇫🇷 Ligue 1": 61, "🇮🇹 Serie A": 135, "🇩🇪 Bundesliga": 78
}

# --- MOTEUR ---
@st.cache_data(ttl=3600)
def get_api(endpoint, params):
    try:
        r = requests.get(f"{BASE_URL}{endpoint}", headers=HEADERS, params=params, timeout=12)
        return r.json().get('response', [])
    except: return []

def calculate_probs(lh, la):
    matrix = np.zeros((10, 10))
    for x in range(10):
        for y in range(10):
            prob = poisson.pmf(x, lh) * poisson.pmf(y, la)
            matrix[x, y] = prob
    matrix /= matrix.sum()
    return {"p_h": np.sum(np.tril(matrix, -1)), "p_n": np.sum(np.diag(matrix)), "p_a": np.sum(np.triu(matrix, 1)), "matrix": matrix}

def get_lambda(team_id, league_id):
    f = get_api("fixtures", {"team": team_id, "season": SEASON, "last": 10})
    if not f: return 1.3
    goals = [(m['goals']['home'] if m['teams']['home']['id'] == team_id else m['goals']['away']) or 0 for m in f]
    w = [0.9**i for i in range(len(goals))]
    return sum(g * weight for g, weight in zip(reversed(goals), w)) / sum(w)

# --- INTERFACE ---
st.title("🏆 iTrOz Predictor V4.8")
tab1, tab2 = st.tabs(["🎯 ANALYSE 1VS1 (BET & AUDIT)", "🚀 TEAM SCANNER (NO-MATCH)"])

with tab1:
    l_name = st.selectbox("LIGUE", [k for k in LEAGUES_DICT.keys() if k != "🌍 TOUS LES CHAMPIONNATS"])
    teams_res = get_api("teams", {"league": LEAGUES_DICT[l_name], "season": SEASON})
    teams = {t['team']['name']: t['team']['id'] for t in teams_res}
    
    if teams:
        c1, c2 = st.columns(2)
        t_h, t_a = c1.selectbox("DOMICILE", sorted(teams.keys())), c2.selectbox("EXTÉRIEUR", sorted(teams.keys()))
        
        if st.button("LANCER L'ANALYSE"):
            lh, la = get_lambda(teams[t_h], LEAGUES_DICT[l_name]), get_lambda(teams[t_a], LEAGUES_DICT[l_name])
            st.session_state.r1v1 = {"res": calculate_probs(lh, la), "th": t_h, "ta": t_a}

    if 'r1v1' in st.session_state:
        d = st.session_state.r1v1
        res, th, ta = d["res"], d["th"], d["ta"]
        
        col1, col2, col3 = st.columns(3)
        col1.metric(th, f"{res['p_h']*100:.1f}%")
        col2.metric("NUL", f"{res['p_n']*100:.1f}%")
        col3.metric(ta, f"{res['p_a']*100:.1f}%")

        # MODE BET & AUDIT
        st.subheader("💰 GESTION BET & AUDIT")
        with st.container():
            st.markdown("<div class='bet-card'>", unsafe_allow_html=True)
            a1, a2, a3 = st.columns([2,1,1])
            pari_user = a1.selectbox("VOTRE SÉLECTION", [th, "Nul", ta, f"{th} ou Nul", f"Nul ou {ta}"])
            cote_user = a2.number_input("COTE DU BOOK", value=1.20)
            capital = a3.number_input("CAPITAL (€)", value=100.0)
            
            p_user = res['p_h'] if pari_user == th else (res['p_n'] if pari_user == "Nul" else res['p_a'])
            if "ou" in pari_user: p_user = (res['p_h']+res['p_n']) if th in pari_user else (res['p_a']+res['p_n'])
            
            ev = p_user * cote_user
            verdict = "🔥 NO-MATCH" if ev > 1.10 and p_user > 0.70 else ("✅ VALABLE" if ev > 1.02 else "❌ DANGEREUX")
            
            # Kelly sécurisé (20% du critère pour préserver la bankroll)
            b = cote_user - 1
            k = max(0, ((b * p_user) - (1 - p_user)) / b) if b > 0 else 0
            
            st.markdown(f"<div class='verdict-box'><b>AUDIT : {verdict}</b><br>EV Indice : {ev:.2f} | Mise conseillée : {(capital*k*0.2):.2f}€</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

with tab2:
    st.subheader("🚀 SCANNER DE NO-MATCH (DÈS @1.12)")
    s1, s2, s3 = st.columns(3)
    l_scan = s1.selectbox("LIGUE", list(LEAGUES_DICT.keys()), key="s_l")
    d_scan = s2.date_input("DATE", datetime.now(), key="s_d")
    m_scan = s3.select_slider("MODE", options=list(ALGO_MODES.keys()), value="ULTRA SAFE (NO-MATCH)")
    
    if st.button("LANCER LE SCAN"):
        cfg = ALGO_MODES[m_scan]
        lids = [LEAGUES_DICT[l_scan]] if LEAGUES_DICT[l_scan] != "ALL" else [140, 39, 2, 61, 135, 78]
        results = []
        
        with st.spinner("Recherche des favoris solides..."):
            for lid in lids:
                fixtures = get_api("fixtures", {"league": lid, "season": SEASON, "date": d_scan.strftime('%Y-%m-%d')})
                for f in fixtures:
                    lh, la = get_lambda(f['teams']['home']['id'], lid), get_lambda(f['teams']['away']['id'], lid)
                    pr = calculate_probs(lh, la)
                    
                    # Test Domicile et Extérieur
                    for side, p_win, val in [("Home", pr['p_h'], "Home"), ("Away", pr['p_a'], "Away")]:
                        if p_win >= cfg['min_p']:
                            odds = get_api("odds", {"fixture": f['fixture']['id']})
                            if odds:
                                for bet in odds[0]['bookmakers'][0]['bets']:
                                    if bet['name'] == "Match Winner":
                                        for v in bet['values']:
                                            cote = float(v['odd'])
                                            if v['value'] == val and cfg['min_cote'] <= cote <= cfg['max_cote']:
                                                results.append({
                                                    "Match": f"{f['teams']['home']['name']} vs {f['teams']['away']['name']}",
                                                    "Pari": f"{v['value']} Win",
                                                    "Confiance": f"{p_win*100:.1f}%",
                                                    "Cote": cote,
                                                    "EV": p_win * cote
                                                })

            final = sorted(results, key=lambda x: x['EV'], reverse=True)[:cfg['max_legs']]
            if final:
                st.success(f"{len(final)} No-Match trouvés !")
                st.table(final)
                st.metric("COTE TOTAL COMBINÉE", f"@{np.prod([x['Cote'] for x in final]):.2f}")
            else:
                st.warning("Aucun No-Match répondant aux critères de sécurité aujourd'hui.")

st.markdown("<div style='text-align:center; opacity:0.2; margin-top:50px;'>iTrOz Predictor v4.8 - Spécialiste No-Match</div>", unsafe_allow_html=True)
