import streamlit as st
import requests
import numpy as np
from scipy.stats import poisson
from datetime import datetime

# --- CONFIGURATION ET STYLE ---
st.set_page_config(page_title="iTrOz Predictor V4.5", layout="wide")

st.markdown("""
    <style>
    .stApp {
        background-image: url("https://media.giphy.com/media/VZrfUvQjXaGEQy1RSn/giphy.gif");
        background-size: cover;
        background-attachment: fixed;
    }
    .stApp > div:first-child { background-color: rgba(0, 0, 0, 0.93); }
    h1, h2, h3, p, span, label { color: #FFD700 !important; font-family: 'Monospace', sans-serif; }
    
    .verdict-box {
        border: 2px solid #FFD700; padding: 20px; text-align: center;
        border-radius: 15px; background: rgba(255, 215, 0, 0.05); margin: 15px 0;
    }
    .stMetric { background: rgba(255, 255, 255, 0.05); padding: 10px; border-radius: 10px; border: 1px solid rgba(255, 215, 0, 0.2); }
    </style>
""", unsafe_allow_html=True)

# --- CONFIG API ---
API_KEY = st.secrets["MY_API_KEY"]
BASE_URL = "https://v3.football.api-sports.io/"
HEADERS = {'x-apisports-key': API_KEY}
SEASON = 2025

# Configuration du Scanner pour favoriser les "No-Match"
ALGO_MODES = {
    "NO-MATCH (ULTRA SAFE)": {"min_p": 0.70, "max_cote": 1.60, "min_ev": 1.05, "kelly": 0.15, "max_legs": 3},
    "CONFIANCE (MID)": {"min_p": 0.55, "max_cote": 2.20, "min_ev": 1.08, "kelly": 0.30, "max_legs": 4},
    "VALUE (AGRESSIF)": {"min_p": 0.40, "max_cote": 3.50, "min_ev": 1.12, "kelly": 0.50, "max_legs": 6}
}

LEAGUES_DICT = {
    "🌍 TOUS LES CHAMPIONNATS": "ALL",
    "🇪🇸 La Liga": 140, "🇬🇧 Premier League": 39, "🇪🇺 Champions League": 2, 
    "🇫🇷 Ligue 1": 61, "🇮🇹 Serie A": 135, "🇩🇪 Bundesliga": 78
}

# --- MOTEUR DE CALCUL ---
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
    f = get_api("fixtures", {"team": team_id, "season": SEASON, "last": 8})
    if not f: return 1.3
    goals = [(m['goals']['home'] if m['teams']['home']['id'] == team_id else m['goals']['away']) or 0 for m in f]
    weights = [0.85**i for i in range(len(goals))]
    return sum(g * w for g, w in zip(reversed(goals), weights)) / sum(weights)

# --- UI PRINCIPALE ---
st.title("🏆 iTrOz Predictor V4.5")
tab1, tab2 = st.tabs(["🎯 ANALYSE 1VS1 (BET & AUDIT)", "🚀 TEAM SCANNER (NO-MATCH)"])

# --- MODE 1V1 : BET & AUDIT ---
with tab1:
    l_choice = st.selectbox("LIGUE", [k for k in LEAGUES_DICT.keys() if k != "🌍 TOUS LES CHAMPIONNATS"])
    teams_data = get_api("teams", {"league": LEAGUES_DICT[l_choice], "season": SEASON})
    teams = {t['team']['name']: t['team']['id'] for t in teams_data}
    
    if teams:
        colA, colB = st.columns(2)
        t_h = colA.selectbox("DOMICILE", sorted(teams.keys()))
        t_a = colB.selectbox("EXTÉRIEUR", sorted(teams.keys()))
        
        if st.button("LANCER L'ANALYSE DÉTAILLÉE"):
            lh, la = get_lambda(teams[t_h], LEAGUES_DICT[l_choice]), get_lambda(teams[t_a], LEAGUES_DICT[l_choice])
            res = calculate_probs(lh, la)
            st.session_state.data = {"res": res, "t_h": t_h, "t_a": t_a}

    if 'data' in st.session_state:
        d = st.session_state.data
        r, th, ta = d["res"], d["t_h"], d["t_a"]
        
        # Affichage Probabilités
        c1, c2, c3 = st.columns(3)
        c1.metric(th, f"{r['p_h']*100:.1f}%")
        c2.metric("NUL", f"{r['p_n']*100:.1f}%")
        c3.metric(ta, f"{r['p_a']*100:.1f}%")

        # --- BLOC MODE BET ---
        st.subheader("💰 MODE BET")
        with st.container():
            b1, b2, b3, b4 = st.columns(4)
            capital = b1.number_input("Capital (€)", value=100.0)
            c_h = b2.number_input(f"Cote {th}", value=2.0)
            c_n = b3.number_input("Cote Nul", value=3.2)
            c_a = b4.number_input(f"Cote {ta}", value=3.5)
            
            opts = [{"n": th, "p": r['p_h'], "c": c_h}, {"n": "Nul", "p": r['p_n'], "c": c_n}, {"n": ta, "p": r['p_a'], "c": c_a}]
            best = max(opts, key=lambda x: x['p'] * x['c'])
            if (best['p'] * best['c']) > 1.05:
                st.info(f"💡 Meilleure Value détectée : **{best['n']}**")

        # --- BLOC AUDIT ---
        st.subheader("🕵️ MODE AUDIT")
        with st.expander("VÉRIFIER VOTRE PROPRE PARI", expanded=True):
            a1, a2 = st.columns(2)
            pari_user = a1.selectbox("Votre sélection", [th, "Nul", ta, f"{th} ou Nul", f"Nul ou {ta}"])
            cote_user = a2.number_input("Cote du bookmaker", value=1.50, key="audit_cote")
            
            # Calcul proba combinée pour l'audit
            p_user = r['p_h'] if pari_user == th else (r['p_n'] if pari_user == "Nul" else r['p_a'])
            if "ou" in pari_user:
                p_user = (r['p_h'] + r['p_n']) if th in pari_user else (r['p_a'] + r['p_n'])
            
            ev_user = p_user * cote_user
            confiance = "🔥 EXCELLENT" if ev_user > 1.15 else ("✅ VALABLE" if ev_user > 1.0 else "❌ DANGEREUX")
            st.markdown(f"<div class='verdict-box'>VERDICT AUDIT : {confiance} (Indice EV : {ev_user:.2f})</div>", unsafe_allow_html=True)

# --- MODE TEAM : SCANNER NO-MATCH ---
with tab2:
    st.subheader("🚀 SCANNER DE MATCHS À HAUTE PROBABILITÉ")
    s1, s2, s3 = st.columns(3)
    l_scan = s1.selectbox("LIGUE À SCANNER", list(LEAGUES_DICT.keys()), key="scan_l")
    d_scan = s2.date_input("DATE", datetime.now(), key="scan_d")
    m_scan = s3.select_slider("MODE ALGO", options=list(ALGO_MODES.keys()), value="NO-MATCH (ULTRA SAFE)")
    
    if st.button("LANCER LE SCAN DES NO-MATCH"):
        cfg = ALGO_MODES[m_scan]
        lids = [LEAGUES_DICT[l_scan]] if LEAGUES_DICT[l_scan] != "ALL" else [140, 39, 2, 61, 135, 78]
        opps = []
        
        with st.spinner("Recherche des No-Match en cours..."):
            for lid in lids:
                fixtures = get_api("fixtures", {"league": lid, "season": SEASON, "date": d_scan.strftime('%Y-%m-%d')})
                for f in fixtures:
                    lh, la = get_lambda(f['teams']['home']['id'], lid), get_lambda(f['teams']['away']['id'], lid)
                    pr = calculate_probs(lh, la)
                    
                    # On ne regarde que les probabilités fortes (> min_p)
                    outcomes = [
                        {"match": f"{f['teams']['home']['name']} Win", "p": pr['p_h'], "val": "Home"},
                        {"match": f"{f['teams']['away']['name']} Win", "p": pr['p_a'], "val": "Away"}
                    ]
                    
                    for o in outcomes:
                        if o['p'] >= cfg['min_p']: # Le filtre No-Match est ici
                            odds = get_api("odds", {"fixture": f['fixture']['id']})
                            if odds:
                                for b in odds[0]['bookmakers'][0]['bets']:
                                    if b['name'] == "Match Winner":
                                        for v in b['values']:
                                            if v['value'] == o['val']:
                                                cote = float(v['odd'])
                                                if cote <= cfg['max_cote']: # Anti-cote folle
                                                    opps.append({"Match": f['teams']['home']['name'] + " vs " + f['teams']['away']['name'], "Pari": o['match'], "Proba IA": f"{o['p']*100:.1f}%", "Cote": cote, "EV": o['p']*cote})

            valid = sorted(opps, key=lambda x: x['EV'], reverse=True)[:cfg['max_legs']]
            if valid:
                cote_total = np.prod([x['Cote'] for x in valid])
                st.success(f"Ticket généré avec {len(valid)} No-Match")
                st.table(valid)
                st.metric("COTE TOTALE DU COMBINÉ", f"@{cote_total:.2f}")
            else:
                st.warning("Aucun No-Match détecté pour cette date. L'IA refuse de prendre des risques inutiles.")

st.markdown("<div style='text-align:center; opacity:0.3; margin-top:50px;'>iTrOz Predictor v4.5 - Intelligence Artificielle et Gestion de Risque</div>", unsafe_allow_html=True)
