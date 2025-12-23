import streamlit as st
import requests
import numpy as np
from scipy.stats import poisson
from datetime import datetime

# --- INTERFACE ET STYLE ---
st.set_page_config(page_title="iTrOz Predictor V5.5 - Infinity", layout="wide")

st.markdown("""
    <style>
    .stApp { background-image: url("https://media.giphy.com/media/VZrfUvQjXaGEQy1RSn/giphy.gif"); background-size: cover; background-attachment: fixed; }
    .stApp > div:first-child { background-color: rgba(0, 0, 0, 0.95); }
    h1, h2, h3, p, span, label { color: #FFD700 !important; font-family: 'Monospace', sans-serif; }
    .verdict-box { border: 2px solid #FFD700; padding: 20px; border-radius: 15px; background: rgba(255, 215, 0, 0.05); text-align: center; margin: 15px 0; }
    .score-card { background: rgba(255, 255, 255, 0.05); border: 1px solid #FFD700; padding: 10px; border-radius: 10px; text-align: center; }
    </style>
""", unsafe_allow_html=True)

# --- CONFIG API ---
API_KEY = st.secrets["MY_API_KEY"]
BASE_URL = "https://v3.football.api-sports.io/"
HEADERS = {'x-apisports-key': API_KEY}
SEASON = 2025

LEAGUES_DICT = {
    "🌍 TOUS LES CHAMPIONNATS": "ALL",
    "🇪🇸 La Liga": 140, "🇬🇧 Premier League": 39, "🇪🇺 Champions League": 2, 
    "🇫🇷 Ligue 1": 61, "🇮🇹 Serie A": 135, "🇩🇪 Bundesliga": 78
}

# --- MOTEUR IA ---
def calculate_probs_v5(lh, la):
    matrix = np.zeros((8, 8))
    for x in range(8):
        for y in range(8):
            matrix[x, y] = poisson.pmf(x, lh) * poisson.pmf(y, la)
    matrix /= matrix.sum()
    
    # Probabilités de base
    p_h = np.sum(np.tril(matrix, -1))
    p_n = np.sum(np.diag(matrix))
    p_a = np.sum(np.triu(matrix, 1))
    
    # BTTS (Les deux marquent)
    p_btts_yes = np.sum(matrix[1:, 1:])
    p_btts_no = 1 - p_btts_yes
    
    return {
        "p_h": p_h, "p_n": p_n, "p_a": p_a,
        "p_1n": p_h + p_n, "p_n2": p_n + p_a, "p_12": p_h + p_a,
        "p_btts": p_btts_yes, "matrix": matrix
    }

def get_lambda(team_id, league_id):
    f = get_api("fixtures", {"team": team_id, "season": SEASON, "last": 10})
    if not f: return 1.3
    goals = [(m['goals']['home'] if m['teams']['home']['id'] == team_id else m['goals']['away']) or 0 for m in f]
    w = [0.9**i for i in range(len(goals))]
    return sum(g * weight for g, weight in zip(reversed(goals), w)) / sum(w)

@st.cache_data(ttl=3600)
def get_api(endpoint, params):
    try:
        r = requests.get(f"{BASE_URL}{endpoint}", headers=HEADERS, params=params, timeout=12)
        return r.json().get('response', [])
    except: return []

# --- APPLICATION ---
st.title("🔱 ITROZ PREDICTOR V5.5")

tab1, tab2 = st.tabs(["🎯 ANALYSE 1VS1 (AUDIT/BET/SCORES)", "🚀 SCANNER MULTI-OPTIONS"])

# --- ONGLET 1 : ANALYSE DÉTAILLÉE ---
with tab1:
    l_name = st.selectbox("LIGUE", [k for k in LEAGUES_DICT.keys() if k != "🌍 TOUS LES CHAMPIONNATS"])
    teams_res = get_api("teams", {"league": LEAGUES_DICT[l_name], "season": SEASON})
    teams = {t['team']['name']: t['team']['id'] for t in teams_res}
    
    if teams:
        c1, c2 = st.columns(2)
        th, ta = c1.selectbox("DOMICILE", sorted(teams.keys())), c2.selectbox("EXTÉRIEUR", sorted(teams.keys()))
        
        if st.button("LANCER L'ANALYSE INFRA-ROUGE"):
            lh, la = get_lambda(teams[th], LEAGUES_DICT[l_name]), get_lambda(teams[ta], LEAGUES_DICT[l_name])
            st.session_state.v55 = {"res": calculate_probs_v5(lh, la), "th": th, "ta": ta}

    if 'v55' in st.session_state:
        r, th, ta = st.session_state.v55["res"], st.session_state.v55["th"], st.session_state.v55["ta"]
        
        # Stats de base
        m1, m2, m3, m4 = st.columns(4)
        m1.metric(th, f"{r['p_h']*100:.1f}%")
        m2.metric("NUL", f"{r['p_n']*100:.1f}%")
        m3.metric(ta, f"{r['p_a']*100:.1f}%")
        m4.metric("BTTS ✅", f"{r['p_btts']*100:.1f}%")

        # --- MODE AUDIT ---
        st.subheader("🕵️ AUDIT DE VOTRE PARI")
        ac1, ac2 = st.columns(2)
        u_bet = ac1.selectbox("VOTRE CHOIX", [th, ta, "Nul", "1N", "N2", "12", "BTTS OUI", "BTTS NON"])
        u_odd = ac2.number_input("COTE DU BOOK", value=1.50, key="aud")
        
        p_map = {th: r['p_h'], ta: r['p_a'], "Nul": r['p_n'], "1N": r['p_1n'], "N2": r['p_n2'], "12": r['p_12'], "BTTS OUI": r['p_btts'], "BTTS NON": 1-r['p_btts']}
        p_final = p_map[u_bet]
        ev = p_final * u_odd
        st.markdown(f"<div class='verdict-box'>INDICE DE FIABILITÉ : {ev:.2f} | {'✅ VALIDÉ' if ev > 1.05 else '❌ RISQUÉ'}</div>", unsafe_allow_html=True)

        # --- MODE BET ---
        st.subheader("💰 MODE BET (OPTIMISATION MISE)")
        bc1, bc2 = st.columns(2)
        cap = bc1.number_input("CAPITAL TOTAL (€)", value=100.0)
        k_mise = max(0, (((u_odd-1)*p_final) - (1-p_final))/(u_odd-1)) if u_odd > 1 else 0
        st.info(f"Mise suggérée (Gestion Pro) : **{(cap * k_mise * 0.2):.2f} €**")

        # --- SCORES PROBABLES ---
        st.subheader("🔢 TOP SCORES PROBABLES")
        idx = np.unravel_index(np.argsort(r['matrix'].ravel())[-5:][::-1], r['matrix'].shape)
        sc = st.columns(5)
        for i in range(5):
            with sc[i]:
                st.markdown(f"<div class='score-card'><b>{idx[0][i]} - {idx[1][i]}</b><br>{r['matrix'][idx[0][i],idx[1][i]]*100:.1f}%</div>", unsafe_allow_html=True)

# --- ONGLET 2 : SCANNER MULTI-OPTIONS ---
with tab2:
    st.subheader("🚀 SCANNER D'OPPORTUNITÉS")
    sc1, sc2, sc3 = st.columns(3)
    l_scan = sc1.selectbox("LIGUE", list(LEAGUES_DICT.keys()), key="lsc")
    d_scan = sc2.date_input("DATE", datetime.now(), key="dsc")
    # LE CURSEUR DE RISQUE
    risk_level = sc3.select_slider("TOLÉRANCE AU RISQUE", options=["ULTRA-SAFE", "MODÉRÉ", "AGRESSIF"], value="MODÉRÉ")
    
    risk_cfg = {
        "ULTRA-SAFE": {"min_p": 0.72, "min_ev": 1.05, "max_legs": 2},
        "MODÉRÉ": {"min_p": 0.58, "min_ev": 1.08, "max_legs": 4},
        "AGRESSIF": {"min_p": 0.45, "min_ev": 1.12, "max_legs": 6}
    }[risk_level]

    if st.button("LANCER LE SCAN DE MARCHÉ"):
        lids = [LEAGUES_DICT[l_scan]] if LEAGUES_DICT[l_scan] != "ALL" else [140, 39, 2, 61, 135, 78]
        opps = []
        
        for lid in lids:
            fixtures = get_api("fixtures", {"league": lid, "season": SEASON, "date": d_scan.strftime('%Y-%m-%d')})
            for f in fixtures:
                lh, la = get_lambda(f['teams']['home']['id'], lid), get_lambda(f['teams']['away']['id'], lid)
                pr = calculate_probs_v5(lh, la)
                
                # On scanne tout : Win, DC, BTTS
                choices = [
                    ("Home", pr['p_h'], "Match Winner", "Home"),
                    ("Away", pr['p_a'], "Match Winner", "Away"),
                    ("Draw", pr['p_n'], "Match Winner", "Draw"),
                    ("Double Chance 1N", pr['p_1n'], "Double Chance", "Home/Draw"),
                    ("Double Chance N2", pr['p_n2'], "Double Chance", "Draw/Away"),
                    ("BTTS YES", pr['p_btts'], "Both Teams Score", "Yes")
                ]
                
                for label, proba, m_name, m_val in choices:
                    if proba >= risk_cfg['min_p']:
                        odds = get_api("odds", {"fixture": f['fixture']['id']})
                        if odds:
                            for b in odds[0]['bookmakers'][0]['bets']:
                                if b['name'] == m_name:
                                    for v in b['values']:
                                        if v['value'] == m_val:
                                            cote = float(v['odd'])
                                            if (proba * cote) >= risk_cfg['min_ev']:
                                                opps.append({
                                                    "Match": f"{f['teams']['home']['name']}-{f['teams']['away']['name']}",
                                                    "Pari": label,
                                                    "Proba": f"{proba*100:.1f}%",
                                                    "Cote": cote,
                                                    "EV": proba*cote
                                                })

        final_ticket = sorted(opps, key=lambda x: x['EV'], reverse=True)[:risk_cfg['max_legs']]
        if final_ticket:
            st.markdown(f"<div class='verdict-box'>TICKET {risk_level} GÉNÉRÉ | @{np.prod([x['Cote'] for x in final_ticket]):.2f}</div>", unsafe_allow_html=True)
            st.table(final_ticket)
        else:
            st.warning("Aucun pari ne correspond à vos critères de risque aujourd'hui.")
