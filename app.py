import streamlit as st
import requests
import numpy as np
from scipy.stats import poisson
from datetime import datetime

# --- CONFIGURATION ET STYLE ---
st.set_page_config(page_title="iTrOz Predictor ULTIMATE V4", layout="wide")

st.markdown("""
    <style>
    .stApp {
        background-image: url("https://media.giphy.com/media/VZrfUvQjXaGEQy1RSn/giphy.gif");
        background-size: cover;
        background-attachment: fixed;
    }
    .stApp > div:first-child { background-color: rgba(0, 0, 0, 0.92); }
    h1, h2, h3, p, span, label { color: #FFD700 !important; font-family: 'Monospace', sans-serif; }
    
    .verdict-text {
        font-size: 22px; font-weight: 900; text-align: center; color: #FFD700;
        border: 1px solid rgba(255, 215, 0, 0.5); padding: 15px; border-radius: 10px;
        background: rgba(255, 215, 0, 0.05); margin: 10px 0;
    }
    .bet-card {
        background: rgba(255, 255, 255, 0.02); padding: 20px; border-radius: 15px;
        border: 1px solid rgba(255, 215, 0, 0.1); margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# --- CONFIGURATION API ---
API_KEY = st.secrets["MY_API_KEY"]
BASE_URL = "https://v3.football.api-sports.io/"
HEADERS = {'x-apisports-key': API_KEY}
SEASON = 2025

ALGO_MODES = {
    "SAFE": {"min_ev": 1.12, "min_p": 0.55, "kelly": 0.15, "max_legs": 2},
    "MID": {"min_ev": 1.08, "min_p": 0.40, "kelly": 0.30, "max_legs": 3},
    "AGRESSIF": {"min_ev": 1.04, "min_p": 0.30, "kelly": 0.50, "max_legs": 5}
}

LEAGUES_DICT = {
    "🌍 TOUS LES CHAMPIONNATS": "ALL",
    "🇪🇸 La Liga": 140, "🇬🇧 Premier League": 39, "🇪🇺 Champions League": 2, 
    "🇫🇷 Ligue 1": 61, "🇮🇹 Serie A": 135, "🇩🇪 Bundesliga": 78
}

@st.cache_data(ttl=3600)
def get_api(endpoint, params):
    try:
        r = requests.get(f"{BASE_URL}{endpoint}", headers=HEADERS, params=params, timeout=12)
        return r.json().get('response', [])
    except: return []

def calculate_probs(lh, la):
    matrix = np.zeros((10, 10))
    # Correction Dixon-Coles simplifiée
    tau = [-0.10, 0.05, 0.05, 0.10] 
    for x in range(10):
        for y in range(10):
            prob = poisson.pmf(x, lh) * poisson.pmf(y, la)
            if x==0 and y==0: prob *= (1 + tau[0])
            elif x==1 and y==1: prob *= (1 + tau[3])
            matrix[x, y] = max(prob, 0)
    matrix /= matrix.sum()
    return {"p_h": np.sum(np.tril(matrix, -1)), "p_n": np.sum(np.diag(matrix)), "p_a": np.sum(np.triu(matrix, 1)), "matrix": matrix}

@st.cache_data(ttl=1800)
def get_lambda(team_id, league_id):
    f = get_api("fixtures", {"team": team_id, "season": SEASON, "last": 8})
    if not f: return 1.35
    goals = [ (m['goals']['home'] if m['teams']['home']['id'] == team_id else m['goals']['away']) or 0 for m in f]
    weights = [0.9**i for i in range(len(goals))]
    return sum(g * w for g, w in zip(reversed(goals), weights)) / sum(weights)

# --- UI ---
st.title("ITROZ PREDICTOR ULTIMATE V4")
tab1, tab2 = st.tabs(["🎯 ANALYSE & AUDIT 1VS1", "🚀 SCANNER DE MARCHÉ"])

with tab1:
    l_name = st.selectbox("LIGUE", [k for k in LEAGUES_DICT.keys() if k != "🌍 TOUS LES CHAMPIONNATS"])
    teams_data = get_api("teams", {"league": LEAGUES_DICT[l_name], "season": SEASON})
    teams = {t['team']['name']: t['team']['id'] for t in teams_data}
    
    if teams:
        c1, c2 = st.columns(2)
        t_h = c1.selectbox("DOMICILE", sorted(teams.keys()))
        t_a = c2.selectbox("EXTÉRIEUR", sorted(teams.keys()))
        
        if st.button("LANCER L'ANALYSE"):
            lh, la = get_lambda(teams[t_h], LEAGUES_DICT[l_name]), get_lambda(teams[t_a], LEAGUES_DICT[l_name])
            res = calculate_probs(lh, la)
            st.session_state.res = res
            st.session_state.t_h, st.session_state.t_a = t_h, t_a

    if 'res' in st.session_state:
        r, th, ta = st.session_state.res, st.session_state.t_h, st.session_state.t_a
        m1, m2, m3 = st.columns(3)
        m1.metric(th, f"{r['p_h']*100:.1f}%")
        m2.metric("NUL", f"{r['p_n']*100:.1f}%")
        m3.metric(ta, f"{r['p_a']*100:.1f}%")

        # --- AUDIT & BET ---
        st.subheader("🔍 AUDIT ET CONSEIL DE MISE")
        st.markdown("<div class='bet-card'>", unsafe_allow_html=True)
        a1, a2, a3 = st.columns([2, 1, 1])
        mon_pari = a1.selectbox("VOTRE PARI POUR AUDIT", [th, "Nul", ta, f"{th} ou Nul", f"Nul ou {ta}", f"{th} ou {ta}"])
        ma_cote = a2.number_input("COTE", value=1.50)
        ma_bankroll = a3.number_input("BANKROLL (€)", value=100.0)

        # Calcul proba audit
        p_audit = r['p_h'] if mon_pari == th else (r['p_n'] if mon_pari == "Nul" else r['p_a'])
        if "ou" in mon_pari:
            if th in mon_pari and "Nul" in mon_pari: p_audit = r['p_h'] + r['p_n']
            elif ta in mon_pari and "Nul" in mon_pari: p_audit = r['p_a'] + r['p_n']
            else: p_audit = r['p_h'] + r['p_a']
        
        ev = p_audit * ma_cote
        status = "✅ SAFE" if ev > 1.10 else ("⚠️ MID" if ev > 0.98 else "❌ DANGEREUX")
        
        # Kelly
        b = ma_cote - 1
        k = max(0, ((b * p_audit) - (1 - p_audit)) / b) if b > 0 else 0
        
        st.markdown(f"<div class='verdict-text'>AUDIT : {status} | EV : {ev:.2f} | MISE OPTIMALE : {(ma_bankroll * k * 0.2):.2f}€</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        with st.expander("🔢 VOIR LES SCORES PROBABLES"):
            idx = np.unravel_index(np.argsort(r['matrix'].ravel())[-5:][::-1], r['matrix'].shape)
            sc = st.columns(5)
            for i in range(5): sc[i].write(f"**{idx[0][i]}-{idx[1][i]}**\n\n{r['matrix'][idx[0][i],idx[1][i]]*100:.1f}%")

with tab2:
    st.subheader("🚀 SCANNER INTELLIGENT")
    s1, s2, s3 = st.columns(3)
    l_scan = s1.selectbox("LIGUE À SCANNER", list(LEAGUES_DICT.keys()))
    d_scan = s2.date_input("DATE", datetime.now())
    m_scan = s3.select_slider("MODE", options=list(ALGO_MODES.keys()), value="MID")
    
    if st.button("LANCER LE SCAN"):
        c = ALGO_MODES[m_scan]
        lids = [LEAGUES_DICT[l_scan]] if LEAGUES_DICT[l_scan] != "ALL" else [140, 39, 2, 61, 135, 78]
        opps = []
        
        for lid in lids:
            fixtures = get_api("fixtures", {"league": lid, "season": SEASON, "date": d_scan.strftime('%Y-%m-%d')})
            for f in fixtures:
                lh, la = get_lambda(f['teams']['home']['id'], lid), get_lambda(f['teams']['away']['id'], lid)
                pr = calculate_probs(lh, la)
                odds = get_api("odds", {"fixture": f['fixture']['id']})
                if odds:
                    for b in odds[0]['bookmakers'][0]['bets']:
                        if b['name'] == "Match Winner":
                            for v in b['values']:
                                p = pr['p_h'] if v['value']=="Home" else (pr['p_n'] if v['value']=="Draw" else pr['p_a'])
                                cote = float(v['odd'])
                                if (p * cote) >= c['min_ev'] and p >= c['min_p']:
                                    opps.append({"Match": f"{f['teams']['home']['name']}-{f['teams']['away']['name']}", "Pari": v['value'], "Cote": cote, "P": p, "EV": p*cote})

        valid = sorted(opps, key=lambda x: x['EV'], reverse=True)[:c['max_legs']]
        if valid:
            ct = np.prod([o['Cote'] for o in valid])
            st.markdown(f"<div class='verdict-text'>TICKET {m_scan} GÉNÉRÉ : @{ct:.2f}</div>", unsafe_allow_html=True)
            st.table(valid)
        else:
            st.error("Aucune opportunité safe trouvée.")
