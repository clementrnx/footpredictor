
import streamlit as st
import requests
import numpy as np
from scipy.stats import poisson
from datetime import datetime

# --- CONFIGURATION ET STYLE FINAL EDITION ---
st.set_page_config(page_title="Clementrnxx Predictor V5.6 ", layout="wide")

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

    .stApp > div:first-child { background-color: rgba(0, 0, 0, 0.92); }
    
    h1, h2, h3, p, span, label { color: #FFD700 !important; font-family: 'Monospace', sans-serif; letter-spacing: 1px; }

    /* Boutons Style Or Premium */
    div.stButton > button {
        background: rgba(255, 215, 0, 0.1) !important;
        backdrop-filter: blur(10px);
        border: 2px solid #FFD700 !important;
        color: #FFD700 !important;
        border-radius: 15px !important;
        letter-spacing: 2px !important;
        font-weight: 900;
        transition: 0.4s;
        width: 100%;
        text-transform: uppercase;
    }
    
    div.stButton > button:hover { 
        background: #FFD700 !important;
        color: black !important;
        box-shadow: 0 0 30px rgba(255, 215, 0, 0.6);
    }

    .verdict-box {
        border: 2px solid #FFD700;
        padding: 25px;
        text-align: center;
        border-radius: 20px;
        background: rgba(0,0,0,0.7);
        margin: 20px 0;
    }

    .score-card {
        background: rgba(255, 255, 255, 0.07);
        border: 1px solid rgba(255, 215, 0, 0.4);
        padding: 15px;
        border-radius: 12px;
        text-align: center;
    }

    .github-link {
        display: block;
        text-align: center;
        color: #FFD700 !important;
        font-weight: bold;
        font-size: 1.2rem;
        text-decoration: none;
        margin-top: 40px;
        padding: 20px;
        border-top: 1px solid rgba(255, 215, 0, 0.2);
    }
    .github-link:hover {
        text-shadow: 0 0 10px #FFD700;
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

# --- FONCTIONS ---
@st.cache_data(ttl=3600)
def get_api(endpoint, params):
    try:
        r = requests.get(f"{BASE_URL}{endpoint}", headers=HEADERS, params=params, timeout=12)
        return r.json().get('response', [])
    except: return []

def calculate_probs(lh, la):
    matrix = np.zeros((8, 8))
    for x in range(8):
        for y in range(8):
            matrix[x, y] = poisson.pmf(x, lh) * poisson.pmf(y, la)
    matrix /= matrix.sum()
    p_h, p_n, p_a = np.sum(np.tril(matrix, -1)), np.sum(np.diag(matrix)), np.sum(np.triu(matrix, 1))
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

# --- UI ---
st.title("🏆 CLEMENTRNXX PREDICTOR ")
st.subheader("V5.6")

tab1, tab2 = st.tabs([" ANALYSE 1VS1", " SCANNER DE TICKETS"])

with tab1:
    l_name = st.selectbox("CHAMPIONNAT", [k for k in LEAGUES_DICT.keys() if k != "TOUS LES CHAMPIONNATS"])
    teams_res = get_api("teams", {"league": LEAGUES_DICT[l_name], "season": SEASON})
    teams = {t['team']['name']: t['team']['id'] for t in teams_res}
    
    if teams:
        c1, c2 = st.columns(2)
        th, ta = c1.selectbox("DOMICILE", sorted(teams.keys())), c2.selectbox("EXTÉRIEUR", sorted(teams.keys()))
        
        if st.button("LANCER L'ANALYSE EXPERTE"):
            lh, la = get_lambda(teams[th], LEAGUES_DICT[l_name]), get_lambda(teams[ta], LEAGUES_DICT[l_name])
            st.session_state.final_cx = {"res": calculate_probs(lh, la), "th": th, "ta": ta}

    if 'final_cx' in st.session_state:
        r, th, ta = st.session_state.final_cx["res"], st.session_state.final_cx["th"], st.session_state.final_cx["ta"]
        
        # PROBABILITES
        m1, m2, m3, m4 = st.columns(4)
        m1.metric(th, f"{r['p_h']*100:.1f}%")
        m2.metric("NUL", f"{r['p_n']*100:.1f}%")
        m3.metric(ta, f"{r['p_a']*100:.1f}%")
        m4.metric("BTTS", f"{r['p_btts']*100:.1f}%")

        # AUDIT DISTINCT
        st.subheader(" AUDIT TECHNIQUE")
        ac1, ac2 = st.columns(2)
        u_bet = ac1.selectbox("VOTRE SÉLECTION", [th, ta, "Nul", "1N", "N2", "12", "BTTS OUI", "BTTS NON"])
        u_odd = ac2.number_input("COTE DU BOOK", value=1.50)
        
        p_map = {th: r['p_h'], ta: r['p_a'], "Nul": r['p_n'], "1N": r['p_1n'], "N2": r['p_n2'], "12": r['p_12'], "BTTS OUI": r['p_btts'], "BTTS NON": 1-r['p_btts']}
        p_final = p_map[u_bet]
        ev = p_final * u_odd
        st.markdown(f"<div class='verdict-box'>INDICE DE FIABILITÉ : {ev:.2f}<br>STATUT : {'✅ VALIDÉ' if ev > 1.05 else '❌ RISQUÉ'}</div>", unsafe_allow_html=True)

        # MODE BET DISTINCT
        st.subheader(" MODE BET")
        bc1, bc2 = st.columns(2)
        cap = bc1.number_input("BANKROLL DISPONIBLE (€)", value=100.0)
        b = u_odd - 1
        kelly = max(0, ((b * p_final) - (1 - p_final)) / b) if b > 0 else 0
        st.info(f"Mise suggérée : **{(cap * kelly * 0.2):.2f} €**")

        # SCORES
        st.subheader("🔢 SCORES PROBABLES")
        idx = np.unravel_index(np.argsort(r['matrix'].ravel())[-5:][::-1], r['matrix'].shape)
        sc = st.columns(5)
        for i in range(5):
            sc[i].markdown(f"<div class='score-card'><b>{idx[0][i]} - {idx[1][i]}</b><br>{r['matrix'][idx[0][i],idx[1][i]]*100:.1f}%</div>", unsafe_allow_html=True)

with tab2:
    st.subheader(" GÉNÉRATEUR DE TICKETS")
    sc1, sc2, sc3 = st.columns(3)
    l_scan = sc1.selectbox("CHAMPIONNAT", list(LEAGUES_DICT.keys()), key="lsc")
    d_scan = sc2.date_input("DATE DU SCAN", datetime.now(), key="dsc")
    
    # CURSEUR DE RISQUE DEMANDÉ
    risk_mode = sc3.select_slider(
        "MODE DE RISQUE", 
        options=["SAFE", "MID-SAFE", "MID", "MID-AGGRESSIF", "AGGRESSIF"],
        value="MID"
    )
    
    risk_cfg = {
        "SAFE": {"p": 0.80, "ev": 1.02, "legs": 2},
        "MID-SAFE": {"p": 0.72, "ev": 1.05, "legs": 3},
        "MID": {"p": 0.62, "ev": 1.08, "legs": 4},
        "MID-AGGRESSIF": {"p": 0.50, "ev": 1.12, "legs": 5},
        "AGGRESSIF": {"p": 0.40, "ev": 1.15, "legs": 7}
    }[risk_mode]

    if st.button("LANCER LE SCAN DES MARCHÉS"):
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
                    ("BTTS OUI", pr['p_btts'], "Both Teams Score", "Yes")
                ]
                
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
                                                opps.append({"MATCH": f"{f['teams']['home']['name']}-{f['teams']['away']['name']}", "PARI": lbl, "PROBA": f"{p*100:.1f}%", "COTE": ct, "VALUE": p*ct})

        res_final = sorted(opps, key=lambda x: x['VALUE'], reverse=True)[:risk_cfg['legs']]
        if res_final:
            st.markdown(f"<div class='verdict-box'>TICKET {risk_mode} GÉNÉRÉ | COTE TOTALE : @{np.prod([x['COTE'] for x in res_final]):.2f}</div>", unsafe_allow_html=True)
            st.table(res_final)
        else:
            st.error("Aucune opportunité trouvée pour ce niveau de risque.")

# --- FOOTER JAUNE ---
st.markdown("""
    <a href="https://github.com/clementrnx" class="github-link" target="_blank">
        GITHUB : github.com/clementrnx
    </a>
    <p style='text-align:center; opacity:0.4; font-size:10px;'>Clementrnxx Predictor V5.6 - Final Edition</p>
""", unsafe_allow_html=True)
