import streamlit as st
import requests
import numpy as np
from scipy.stats import poisson
from datetime import datetime

# --- CONFIGURATION ET STYLE ---
st.set_page_config(page_title="Clementrnxx Predictor V5.5", layout="wide")

st.markdown("""
    <style>
    .stApp {
        background-image: url("https://media.giphy.com/media/VZrfUvQjXaGEQy1RSn/giphy.gif");
        background-size: cover;
        background-attachment: fixed;
    }
    .stApp > div:first-child { background-color: rgba(0, 0, 0, 0.96); }
    h1, h2, h3, p, span, label { color: #FFD700 !important; font-family: 'Courier New', monospace; }
    
    .verdict-box {
        border: 1px solid #FFD700; padding: 20px; text-align: center;
        border-radius: 5px; background: rgba(255, 215, 0, 0.02); margin: 15px 0;
    }
    .score-grid {
        display: grid; grid-template-columns: repeat(5, 1fr); gap: 10px; margin-top: 10px;
    }
    .score-item {
        border: 1px solid rgba(255, 215, 0, 0.3); padding: 10px; text-align: center; background: rgba(255,255,255,0.05);
    }
    div.stButton > button {
        background: transparent !important; border: 1px solid #FFD700 !important;
        color: #FFD700 !important; border-radius: 0px !important; width: 100%;
        text-transform: uppercase; letter-spacing: 2px;
    }
    </style>
""", unsafe_allow_html=True)

# --- CONFIG API ---
API_KEY = st.secrets["MY_API_KEY"]
BASE_URL = "https://v3.football.api-sports.io/"
HEADERS = {'x-apisports-key': API_KEY}
SEASON = 2025

LEAGUES_DICT = {
    "TOUS LES CHAMPIONNATS": "ALL",
    "La Liga": 140, "Premier League": 39, "Champions League": 2, 
    "Ligue 1": 61, "Serie A": 135, "Bundesliga": 78
}

# --- LOGIQUE TECHNIQUE ---
def calculate_advanced_probs(lh, la):
    matrix = np.zeros((8, 8))
    for x in range(8):
        for y in range(8):
            matrix[x, y] = poisson.pmf(x, lh) * poisson.pmf(y, la)
    matrix /= matrix.sum()
    
    p_h, p_n, p_a = np.sum(np.tril(matrix, -1)), np.sum(np.diag(matrix)), np.sum(np.triu(matrix, 1))
    p_btts = np.sum(matrix[1:, 1:])
    
    return {
        "p_h": p_h, "p_n": p_n, "p_a": p_a,
        "p_1n": p_h + p_n, "p_n2": p_n + p_a, "p_12": p_h + p_a,
        "p_btts": p_btts, "matrix": matrix
    }

def get_lambda_v5(team_id, league_id):
    f = get_api("fixtures", {"team": team_id, "season": SEASON, "last": 10})
    if not f: return 1.3
    goals = [(m['goals']['home'] if m['teams']['home']['id'] == team_id else m['goals']['away']) or 0 for m in f]
    weights = [0.9**i for i in range(len(goals))]
    return sum(g * w for g, w in zip(reversed(goals), weights)) / sum(weights)

@st.cache_data(ttl=3600)
def get_api(endpoint, params):
    try:
        r = requests.get(f"{BASE_URL}{endpoint}", headers=HEADERS, params=params, timeout=12)
        return r.json().get('response', [])
    except: return []

# --- INTERFACE ---
st.title("CLEMENTRNXX PREDICTOR V5.5")

tab1, tab2 = st.tabs(["ANALYSE DÉTAILLÉE 1VS1", "SCANNER DE MARCHÉ"])

with tab1:
    l_name = st.selectbox("CHAMPIONNAT", [k for k in LEAGUES_DICT.keys() if k != "TOUS LES CHAMPIONNATS"])
    teams_res = get_api("teams", {"league": LEAGUES_DICT[l_name], "season": SEASON})
    teams = {t['team']['name']: t['team']['id'] for t in teams_res}
    
    if teams:
        c1, c2 = st.columns(2)
        th, ta = c1.selectbox("DOMICILE", sorted(teams.keys())), c2.selectbox("EXTÉRIEUR", sorted(teams.keys()))
        
        if st.button("EXECUTER L'ANALYSE"):
            lh, la = get_lambda_v5(teams[th], LEAGUES_DICT[l_name]), get_lambda_v5(teams[ta], LEAGUES_DICT[l_name])
            st.session_state.cx = {"res": calculate_advanced_probs(lh, la), "th": th, "ta": ta}

    if 'cx' in st.session_state:
        r, th, ta = st.session_state.cx["res"], st.session_state.cx["th"], st.session_state.cx["ta"]
        
        m1, m2, m3, m4 = st.columns(4)
        m1.metric(th, f"{r['p_h']*100:.1f}%")
        m2.metric("NUL", f"{r['p_n']*100:.1f}%")
        m3.metric(ta, f"{r['p_a']*100:.1f}%")
        m4.metric("BTTS", f"{r['p_btts']*100:.1f}%")

        # MODE AUDIT
        st.subheader("AUDIT TECHNIQUE")
        ac1, ac2 = st.columns(2)
        u_bet = ac1.selectbox("SELECTION", [th, ta, "Nul", "1N", "N2", "12", "BTTS OUI", "BTTS NON"])
        u_odd = ac2.number_input("COTE BOOKMAKER", value=1.50)
        
        p_map = {th: r['p_h'], ta: r['p_a'], "Nul": r['p_n'], "1N": r['p_1n'], "N2": r['p_n2'], "12": r['p_12'], "BTTS OUI": r['p_btts'], "BTTS NON": 1-r['p_btts']}
        p_val = p_map[u_bet]
        ev = p_val * u_odd
        st.markdown(f"<div class='verdict-box'>INDICE EV : {ev:.2f} | STATUT : {'VALIDE' if ev > 1.05 else 'RISQUE ELEVE'}</div>", unsafe_allow_html=True)

        # MODE BET
        st.subheader("GESTION DES MISES (MODE BET)")
        bc1, bc2 = st.columns(2)
        capital = bc1.number_input("CAPITAL DISPONIBLE", value=100.0)
        b_coeff = u_odd - 1
        kelly = max(0, ((b_coeff * p_val) - (1 - p_val)) / b_coeff) if b_coeff > 0 else 0
        st.write(f"MISE OPTIMISEE : {(capital * kelly * 0.25):.2f} units")

        # SCORES
        st.subheader("SCORES PROBABLES")
        idx = np.unravel_index(np.argsort(r['matrix'].ravel())[-5:][::-1], r['matrix'].shape)
        sc_cols = st.columns(5)
        for i in range(5):
            sc_cols[i].markdown(f"<div class='score-item'><b>{idx[0][i]} - {idx[1][i]}</b><br>{r['matrix'][idx[0][i],idx[1][i]]*100:.1f}%</div>", unsafe_allow_html=True)

with tab2:
    st.subheader("SCANNER HAUT RENDEMENT")
    sc1, sc2, sc3 = st.columns(3)
    l_scan = sc1.selectbox("LIGUE CIBLE", list(LEAGUES_DICT.keys()), key="lsc")
    d_scan = sc2.date_input("DATE ANALYSE", datetime.now(), key="dsc")
    
    # CURSEUR DE RISQUE ETENDU
    risk_level = sc3.select_slider(
        "TOLERANCE RISQUE", 
        options=["BANQUE", "SECURITE", "EQUILIBRE", "OFFENSIF", "JACKPOT"],
        value="EQUILIBRE"
    )
    
    risk_map = {
        "BANQUE": {"min_p": 0.80, "min_ev": 1.02, "legs": 2},
        "SECURITE": {"min_p": 0.70, "min_ev": 1.05, "legs": 3},
        "EQUILIBRE": {"min_p": 0.58, "min_ev": 1.08, "legs": 4},
        "OFFENSIF": {"min_p": 0.45, "min_ev": 1.12, "legs": 5},
        "JACKPOT": {"min_p": 0.35, "min_ev": 1.15, "legs": 8}
    }

    if st.button("LANCER LE SCAN"):
        cfg = risk_map[risk_level]
        lids = [LEAGUES_DICT[l_scan]] if LEAGUES_DICT[l_scan] != "ALL" else [140, 39, 2, 61, 135, 78]
        opps = []
        
        for lid in lids:
            fixtures = get_api("fixtures", {"league": lid, "season": SEASON, "date": d_scan.strftime('%Y-%m-%d')})
            for f in fixtures:
                lh, la = get_lambda_v5(f['teams']['home']['id'], lid), get_lambda_v5(f['teams']['away']['id'], lid)
                pr = calculate_advanced_probs(lh, la)
                
                tests = [
                    ("1", pr['p_h'], "Match Winner", "Home"),
                    ("2", pr['p_a'], "Match Winner", "Away"),
                    ("1N", pr['p_1n'], "Double Chance", "Home/Draw"),
                    ("N2", pr['p_n2'], "Double Chance", "Draw/Away"),
                    ("BTTS", pr['p_btts'], "Both Teams Score", "Yes")
                ]
                
                for label, proba, m_name, m_val in tests:
                    if proba >= cfg['min_p']:
                        odds = get_api("odds", {"fixture": f['fixture']['id']})
                        if odds:
                            for b in odds[0]['bookmakers'][0]['bets']:
                                if b['name'] == m_name:
                                    for v in b['values']:
                                        if v['value'] == m_val:
                                            cote = float(v['odd'])
                                            if (proba * cote) >= cfg['min_ev']:
                                                opps.append({
                                                    "MATCH": f"{f['teams']['home']['name']} - {f['teams']['away']['name']}",
                                                    "TYPE": label,
                                                    "PROBA": f"{proba*100:.1f}%",
                                                    "COTE": cote,
                                                    "POWER": proba * cote
                                                })

        ticket = sorted(opps, key=lambda x: x['POWER'], reverse=True)[:cfg['legs']]
        if ticket:
            st.markdown(f"<div class='verdict-box'>TICKET {risk_level} GENERE | COTE FINALE : @{np.prod([x['COTE'] for x in ticket]):.2f}</div>", unsafe_allow_html=True)
            st.table(ticket)
        else:
            st.write("AUCUNE OPPORTUNITE DETECTEE POUR CE NIVEAU DE RISQUE.")

st.markdown("<div style='text-align:center; opacity:0.1; margin-top:50px;'>CLEMENTRNXX PREDICTOR V5.5 - NO EMOJI EDITION</div>", unsafe_allow_html=True)
