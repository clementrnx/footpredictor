import streamlit as st
import requests
import numpy as np
from scipy.stats import poisson
from datetime import datetime, timedelta

# --- CONFIGURATION ET STYLE ---
st.set_page_config(page_title="iTrOz Predictor PRO", layout="wide")

st.markdown("""
    <style>
    @keyframes subtleDistort {
        0% { transform: scale(1.0); filter: hue-rotate(0deg) brightness(1); }
        50% { transform: scale(1.01) contrast(1.1); filter: hue-rotate(2deg) brightness(1.1); }
        100% { transform: scale(1.0); filter: hue-rotate(0deg) brightness(1); }
    }

    .stApp {
        background-image: url("https://media.giphy.com/media/VZrfUvQjXaGEQy1RSn/giphy.gif");
        background-size: cover;
        background-attachment: fixed;
        animation: subtleDistort 15s infinite ease-in-out;
    }

    .stApp > div:first-child { background-color: rgba(0, 0, 0, 0.88); }
    
    h1, h2, h3, p, span, label { color: #FFD700 !important; font-family: 'Monospace', sans-serif; letter-spacing: 1px; }

    /* Boutons et Inputs style Or */
    div.stButton > button {
        background: rgba(255, 215, 0, 0.05) !important;
        backdrop-filter: blur(15px);
        border: 1px solid rgba(255, 215, 0, 0.3) !important;
        color: #FFD700 !important;
        border-radius: 12px !important;
        letter-spacing: 5px !important;
        transition: 0.4s;
        width: 100%;
    }
    
    div.stButton > button:hover { 
        background: rgba(255, 215, 0, 0.2) !important;
        box-shadow: 0 0 20px rgba(255, 215, 0, 0.3);
    }

    .verdict-box {
        border: 2px solid #FFD700;
        padding: 20px;
        text-align: center;
        border-radius: 15px;
        background: rgba(0,0,0,0.5);
        margin: 20px 0;
    }
    </style>
""", unsafe_allow_html=True)

# --- CONFIG API ---
API_KEY = st.secrets["MY_API_KEY"]
BASE_URL = "https://v3.football.api-sports.io/"
HEADERS = {'x-apisports-key': API_KEY}
SEASON = 2025

# --- MODES DE L'ALGO (V1) ---
ALGO_MODES = {
    "SAFE": {"min_ev": 1.15, "kelly": 0.10, "max_legs": 2, "color": "#00FF7F"},
    "MID": {"min_ev": 1.07, "kelly": 0.35, "max_legs": 4, "color": "#FFD700"},
    "AGRESSIF": {"min_ev": 1.02, "kelly": 0.75, "max_legs": 8, "color": "#FF4500"},
    "FOU": {"min_ev": 0.95, "kelly": 1.00, "max_legs": 15, "color": "#FF0000"}
}

# --- FONCTIONS CALCULS (V2) ---
@st.cache_data(ttl=3600)
def get_api(endpoint, params):
    try:
        r = requests.get(f"{BASE_URL}{endpoint}", headers=HEADERS, params=params, timeout=12)
        return r.json().get('response', [])
    except: return []

def calculate_dixon_coles(lh, la):
    tau = [-0.13, 0.065, 0.065, 0.13]
    matrix = np.zeros((8, 8))
    for x in range(8):
        for y in range(8):
            prob = poisson.pmf(x, lh) * poisson.pmf(y, la)
            if x==0 and y==0: prob *= (1 + tau[0]*lh*la)
            elif x==1 and y==0: prob *= (1 + tau[1]*lh)
            elif x==0 and y==1: prob *= (1 + tau[2]*la)
            elif x==1 and y==1: prob *= (1 + tau[3])
            matrix[x, y] = max(prob, 0)
    matrix /= matrix.sum()
    return {
        'p_h': np.sum(np.tril(matrix, -1)),
        'p_n': np.sum(np.diag(matrix)),
        'p_a': np.sum(np.triu(matrix, 1)),
        'matrix': matrix
    }

# --- LOGIQUE XG PONDÉRÉE (V2) ---
def get_team_lambda(team_id, league_id):
    fixtures = get_api("fixtures", {"team": team_id, "league": league_id, "season": SEASON, "last": 8})
    if not fixtures: return 1.3 # Fallback
    
    weighted_goals = 0
    total_weight = 0
    for i, f in enumerate(reversed(fixtures)):
        weight = 0.9 ** i # Décroissance temporelle
        goals = f['goals']['home'] if f['teams']['home']['id'] == team_id else f['goals']['away']
        weighted_goals += (goals or 0) * weight
        total_weight += weight
    return weighted_goals / total_weight

# --- INTERFACE ---
st.title("🏆 ITROZ PREDICTOR PRO")

tab1, tab2 = st.tabs(["🎯 ANALYSE 1VS1", "🔍 ALGO SCANNER"])

with tab1:
    leagues = {"La Liga": 140, "Premier League": 39, "Champions League": 2, "Serie A": 135, "Ligue 1": 61}
    l_id = leagues[st.selectbox("LIGUE", list(leagues.keys()), key="l1")]
    
    teams_res = get_api("teams", {"league": l_id, "season": SEASON})
    teams = {t['team']['name']: t['team']['id'] for t in teams_res}
    
    if teams:
        c1, c2 = st.columns(2)
        t_h = c1.selectbox("DOMICILE", sorted(teams.keys()))
        t_a = c2.selectbox("EXTÉRIEUR", sorted(teams.keys()))
        
        if st.button("LANCER L'ANALYSE"):
            lh, la = get_team_lambda(teams[t_h], l_id), get_team_lambda(teams[t_a], l_id)
            res = calculate_dixon_coles(lh, la)
            
            st.markdown(f"<div class='verdict-box'><h3>PROBABILITÉS IA</h3>"
                        f"{t_h} : {res['p_h']*100:.1f}% | NUL : {res['p_n']*100:.1f}% | {t_a} : {res['p_a']*100:.1f}%</div>", 
                        unsafe_allow_html=True)

with tab2:
    st.subheader("SCAN AUTOMATIQUE DU MARCHÉ")
    col_sc1, col_sc2 = st.columns(2)
    scan_date = col_sc1.date_input("DATE", datetime.now())
    mode_name = col_sc2.select_slider("TEMPÉRAMENT DE L'ALGO", options=list(ALGO_MODES.keys()), value="MID")
    
    bankroll = st.number_input("BANKROLL TOTAL (€)", value=100.0)
    conf = ALGO_MODES[mode_name]

    if st.button("🚀 EXÉCUTER L'ALGO SUR LA JOURNÉE"):
        with st.spinner("Analyse Dixon-Coles en cours sur tous les matchs..."):
            fixtures = get_api("fixtures", {"league": l_id, "season": SEASON, "date": scan_date.strftime('%Y-%m-%d')})
            all_opps = []

            for f in fixtures:
                f_id = f['fixture']['id']
                lh = get_team_lambda(f['teams']['home']['id'], l_id)
                la = get_team_lambda(f['teams']['away']['id'], l_id)
                probs = calculate_dixon_coles(lh, la)
                
                odds_res = get_api("odds", {"fixture": f_id})
                if not odds_res: continue
                
                for bet in odds_res[0]['bookmakers'][0]['bets']:
                    if bet['name'] == "Match Winner":
                        for v in bet['values']:
                            p = probs['p_h'] if v['value']=="Home" else (probs['p_n'] if v['value']=="Draw" else probs['p_a'])
                            cote = float(v['odd'])
                            if (p * cote) >= conf['min_ev']:
                                all_opps.append({
                                    "Match": f"{f['teams']['home']['name']} - {f['teams']['away']['name']}",
                                    "Pari": v['value'],
                                    "Cote": cote,
                                    "EV": round(p * cote, 2),
                                    "Proba": p
                                })

            # --- GÉNÉRATION TICKET ---
            valid_opps = sorted(all_opps, key=lambda x: x['EV'], reverse=True)[:conf['max_legs']]
            if valid_opps:
                cote_t = np.prod([o['Cote'] for o in valid_opps])
                prob_t = np.prod([o['Proba'] for o in valid_opps])
                
                # Kelly
                b = cote_t - 1
                k_mise = ((b * prob_t - (1 - prob_t)) / b) * conf['kelly'] if b > 0 else 0
                mise_f = max(0, bankroll * k_mise)

                st.markdown(f"<div class='verdict-box'><h2 style='color:#FFD700'>TICKET GÉNÉRÉ : @{cote_t:.2f}</h2>"
                            f"MISE CONSEILLÉE : {mise_f:.2f}€</div>", unsafe_allow_html=True)
                st.table(valid_opps)
            else:
                st.warning("Aucune Value détectée avec ces paramètres.")

st.markdown("<div style='text-align:center; margin-top:50px; opacity:0.3;'>iTrOz Predictor v3.5 - L'ALGO PRO</div>", unsafe_allow_html=True)
