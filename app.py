import streamlit as st
import requests
import numpy as np
from scipy.stats import poisson
from datetime import datetime

# --- CONFIGURATION ET STYLE FINAL EDITION ---
st.set_page_config(page_title="Clementrnxx Predictor V5.5 ", layout="wide")

st.markdown("""
    <style>
    @keyframes subtleDistort {
        0% { transform: scale(1.0); filter: brightness(1); }
        50% { transform: scale(1.01) brightness(1.1); }
        100% { transform: scale(1.0); filter: brightness(1); }
    }

    .stApp {
        background-image: url("https://media.giphy.com/media/VZrfUvQjXaGEQy1RSn/giphy.gif");
        background-size: cover;
        background-attachment: fixed;
        animation: subtleDistort 15s infinite ease-in-out;
    }

    .stApp > div:first-child { background-color: rgba(0, 0, 0, 0.9); }
    
    h1, h2, h3, p, span, label { color: #FFD700 !important; font-family: 'Monospace', sans-serif; letter-spacing: 1px; }

    /* Boutons Style Or Premium */
    div.stButton > button {
        background: rgba(255, 215, 0, 0.08) !important;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 215, 0, 0.4) !important;
        color: #FFD700 !important;
        border-radius: 12px !important;
        letter-spacing: 3px !important;
        font-weight: bold;
        transition: 0.4s;
        width: 100%;
        text-transform: uppercase;
    }
    
    div.stButton > button:hover { 
        background: rgba(255, 215, 0, 0.2) !important;
        box-shadow: 0 0 20px rgba(255, 215, 0, 0.4);
        transform: translateY(-2px);
    }

    .verdict-box {
        border: 2px solid #FFD700;
        padding: 20px;
        text-align: center;
        border-radius: 15px;
        background: rgba(0,0,0,0.6);
        margin: 20px 0;
    }

    .score-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 215, 0, 0.3);
        padding: 10px;
        border-radius: 10px;
        text-align: center;
    }

    .github-btn {
        display: block;
        width: 250px;
        margin: 50px auto;
        padding: 15px;
        text-align: center;
        background: rgba(255, 215, 0, 0.1);
        border: 1px solid #FFD700;
        color: #FFD700;
        text-decoration: none;
        border-radius: 30px;
        font-weight: bold;
        transition: 0.3s;
    }
    .github-btn:hover {
        background: #FFD700;
        color: black;
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

# --- FONCTIONS TECHNIQUES ---
def calculate_probs(lh, la):
    matrix = np.zeros((8, 8))
    for x in range(8):
        for y in range(8):
            matrix[x, y] = poisson.pmf(x, lh) * poisson.pmf(y, la)
    matrix /= matrix.sum()
    
    p_h = np.sum(np.tril(matrix, -1))
    p_n = np.sum(np.diag(matrix))
    p_a = np.sum(np.triu(matrix, 1))
    
    return {
        "p_h": p_h, "p_n": p_n, "p_a": p_a,
        "p_1n": p_h + p_n, "p_n2": p_n + p_a, "p_12": p_h + p_a,
        "p_btts": np.sum(matrix[1:, 1:]), "matrix": matrix
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

# --- INTERFACE PRINCIPALE ---
st.title("🏆 CLEMENTRNXX PREDICTOR")
st.subheader("V5.5")

tab1, tab2 = st.tabs(["🎯 ANALYSE 1VS1", "🚀 SCANNER MULTI-MODES"])

with tab1:
    l_name = st.selectbox("LIGUE", [k for k in LEAGUES_DICT.keys() if k != "TOUS LES CHAMPIONNATS"])
    teams_res = get_api("teams", {"league": LEAGUES_DICT[l_name], "season": SEASON})
    teams = {t['team']['name']: t['team']['id'] for t in teams_res}
    
    if teams:
        c1, c2 = st.columns(2)
        th, ta = c1.selectbox("DOMICILE", sorted(teams.keys())), c2.selectbox("EXTÉRIEUR", sorted(teams.keys()))
        
        if st.button("LANCER L'ANALYSE"):
            lh, la = get_lambda(teams[th], LEAGUES_DICT[l_name]), get_lambda(teams[ta], LEAGUES_DICT[l_name])
            st.session_state.final = {"res": calculate_probs(lh, la), "th": th, "ta": ta}

    if 'final' in st.session_state:
        r, th, ta = st.session_state.final["res"], st.session_state.final["th"], st.session_state.final["ta"]
        
        m1, m2, m3, m4 = st.columns(4)
        m1.metric(th, f"{r['p_h']*100:.1f}%")
        m2.metric("NUL", f"{r['p_n']*100:.1f}%")
        m3.metric(ta, f"{r['p_a']*100:.1f}%")
        m4.metric("BTTS", f"{r['p_btts']*100:.1f}%")

        # MODE AUDIT
        st.subheader("🕵️ AUDIT DU PARI")
        ac1, ac2 = st.columns(2)
        u_bet = ac1.selectbox("VOTRE CHOIX", [th, ta, "Nul", "1N", "N2", "12", "BTTS OUI", "BTTS NON"])
        u_odd = ac2.number_input("COTE BOOKMAKER", value=1.50)
        
        p_map = {th: r['p_h'], ta: r['p_a'], "Nul": r['p_n'], "1N": r['p_1n'], "N2": r['p_n2'], "12": r['p_12'], "BTTS OUI": r['p_btts'], "BTTS NON": 1-r['p_btts']}
        ev = p_map[u_bet] * u_odd
        st.markdown(f"<div class='verdict-box'>INDICE EV : {ev:.2f} | {'✅ VALIDE' if ev > 1.05 else '⚠️ RISQUÉ'}</div>", unsafe_allow_html=True)

        # MODE BET
        st.subheader("💰 MODE BET (MISES)")
        bc1, bc2 = st.columns(2)
        cap = bc1.number_input("BANKROLL (€)", value=100.0)
        b = u_odd - 1
        k = max(0, ((b * p_map[u_bet]) - (1 - p_map[u_bet])) / b) if b > 0 else 0
        st.info(f"Mise optimale conseillée : **{(cap * k * 0.2):.2f} €**")

        # SCORES
        st.subheader("🔢 SCORES LES PLUS PROBABLES")
        idx = np.unravel_index(np.argsort(r['matrix'].ravel())[-5:][::-1], r['matrix'].shape)
        sc = st.columns(5)
        for i in range(5):
            sc[i].markdown(f"<div class='score-card'><b>{idx[0][i]} - {idx[1][i]}</b><br>{r['matrix'][idx[0][i],idx[1][i]]*100:.1f}%</div>", unsafe_allow_html=True)

with tab2:
    st.subheader("🔍 SCANNER DE MARCHÉ")
    sc1, sc2, sc3 = st.columns(3)
    l_scan = sc1.selectbox("LIGUE", list(LEAGUES_DICT.keys()), key="lsc")
    d_scan = sc2.date_input("DATE", datetime.now(), key="dsc")
    risk_level = sc3.select_slider("MODE DE RISQUE", options=["BANQUE", "SECURITÉ", "EQUILIBRÉ", "OFFENSIF", "JACKPOT"], value="EQUILIBRÉ")
    
    risk_map = {
        "BANQUE": {"p": 0.80, "ev": 1.02, "legs": 2},
        "SECURITÉ": {"p": 0.70, "ev": 1.05, "legs": 3},
        "EQUILIBRÉ": {"p": 0.58, "ev": 1.08, "legs": 4},
        "OFFENSIF": {"p": 0.45, "ev": 1.12, "legs": 5},
        "JACKPOT": {"p": 0.35, "ev": 1.15, "legs": 8}
    }

    if st.button("LANCER LE SCAN DES OPPORTUNITÉS"):
        cfg = risk_map[risk_level]
        lids = [LEAGUES_DICT[l_scan]] if LEAGUES_DICT[l_scan] != "ALL" else [140, 39, 2, 61, 135, 78]
        opps = []
        
        for lid in lids:
            fixtures = get_api("fixtures", {"league": lid, "season": SEASON, "date": d_scan.strftime('%Y-%m-%d')})
            for f in fixtures:
                lh, la = get_lambda(f['teams']['home']['id'], lid), get_lambda(f['teams']['away']['id'], lid)
                pr = calculate_probs(lh, la)
                
                tests = [
                    ("Victoire 1", pr['p_h'], "Match Winner", "Home"),
                    ("Victoire 2", pr['p_a'], "Match Winner", "Away"),
                    ("Double Chance 1N", pr['p_1n'], "Double Chance", "Home/Draw"),
                    ("Double Chance N2", pr['p_n2'], "Double Chance", "Draw/Away"),
                    ("BTTS YES", pr['p_btts'], "Both Teams Score", "Yes")
                ]
                
                for lbl, p, m_n, m_v in tests:
                    if p >= cfg['p']:
                        odds = get_api("odds", {"fixture": f['fixture']['id']})
                        if odds:
                            for btt in odds[0]['bookmakers'][0]['bets']:
                                if btt['name'] == m_n:
                                    for o in btt['values']:
                                        if o['value'] == m_v:
                                            ct = float(o['odd'])
                                            if (p * ct) >= cfg['ev']:
                                                opps.append({"MATCH": f"{f['teams']['home']['name']} - {f['teams']['away']['name']}", "PARI": lbl, "PROBA": f"{p*100:.1f}%", "COTE": ct, "VALUE": p*ct})

        res_final = sorted(opps, key=lambda x: x['VALUE'], reverse=True)[:cfg['legs']]
        if res_final:
            st.markdown(f"<div class='verdict-box'>TICKET {risk_level} GÉNÉRÉ | COTE : @{np.prod([x['COTE'] for x in res_final]):.2f}</div>", unsafe_allow_html=True)
            st.table(res_final)
        else:
            st.error("Aucune opportunité trouvée pour ce niveau de risque.")

# --- FOOTER ---
st.markdown("""
    <a href="https://github.com/clementrnx" class="github-btn" target="_blank">
        MON GITHUB : CLEMENTRNX
    </a>
    <div style='text-align:center; opacity:0.3; padding-bottom:30px;'>
        Clementrnxx Predictor V5.5 - Final Edition
    </div>
""", unsafe_allow_html=True)
