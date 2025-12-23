import streamlit as st
import requests
import numpy as np
from scipy.stats import poisson
from datetime import datetime, timedelta

# --- CONFIGURATION ET STYLE ---
st.set_page_config(page_title="iTrOz Predictor ULTIMATE", layout="wide")

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
    h1, h2, h3, p, span, label { color: #FFD700 !important; font-family: 'Monospace', sans-serif; }
    
    .verdict-text {
        font-size: 24px; font-weight: 900; text-align: center; color: #FFD700;
        border: 2px solid rgba(255, 215, 0, 0.4); padding: 20px; border-radius: 15px;
        background: rgba(0,0,0,0.6); margin: 15px 0;
    }
    .bet-card {
        background: rgba(255, 255, 255, 0.03); padding: 25px; border-radius: 15px;
        border: 1px solid rgba(255, 215, 0, 0.1); margin-bottom: 20px;
    }
    div.stButton > button {
        background: rgba(255, 215, 0, 0.1) !important; color: #FFD700 !important;
        border: 1px solid #FFD700 !important; letter-spacing: 3px; font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# --- CONFIGURATION API & MODES ---
API_KEY = st.secrets["MY_API_KEY"]
BASE_URL = "https://v3.football.api-sports.io/"
HEADERS = {'x-apisports-key': API_KEY}
SEASON = 2025
DISCORD_WEBHOOK = "VOTRE_WEBHOOK_ICI" # À remplir si besoin

ALGO_MODES = {
    "SAFE": {"min_ev": 1.15, "kelly": 0.10, "max_legs": 2, "color": "#00FF7F"},
    "MID": {"min_ev": 1.07, "kelly": 0.35, "max_legs": 4, "color": "#FFD700"},
    "AGRESSIF": {"min_ev": 1.02, "kelly": 0.75, "max_legs": 8, "color": "#FF4500"},
    "FOU": {"min_ev": 0.95, "kelly": 1.00, "max_legs": 15, "color": "#FF0000"}
}

LEAGUES_DICT = {
    "🌍 TOUS LES CHAMPIONNATS": "ALL",
    "🇪🇸 La Liga": 140, 
    "🇬🇧 Premier League": 39, 
    "🇪🇺 Champions League": 2, 
    "🇫🇷 Ligue 1": 61, 
    "🇮🇹 Serie A": 135, 
    "🇩🇪 Bundesliga": 78
}

# --- FONCTIONS TECHNIQUES ---
@st.cache_data(ttl=3600)
def get_api(endpoint, params):
    try:
        r = requests.get(f"{BASE_URL}{endpoint}", headers=HEADERS, params=params, timeout=12)
        return r.json().get('response', [])
    except: return []

def calculate_probs_advanced(lh, la):
    tau = [-0.13, 0.065, 0.065, 0.13]
    matrix = np.zeros((10, 10))
    for x in range(10):
        for y in range(10):
            prob = poisson.pmf(x, lh) * poisson.pmf(y, la)
            if x==0 and y==0: prob *= (1 + tau[0]*lh*la)
            elif x==1 and y==0: prob *= (1 + tau[1]*lh)
            elif x==0 and y==1: prob *= (1 + tau[2]*la)
            elif x==1 and y==1: prob *= (1 + tau[3])
            matrix[x, y] = max(prob, 0)
    matrix /= matrix.sum()
    return {
        "p_h": np.sum(np.tril(matrix, -1)),
        "p_n": np.sum(np.diag(matrix)),
        "p_a": np.sum(np.triu(matrix, 1)),
        "matrix": matrix
    }

@st.cache_data(ttl=1800)
def get_team_lambda_advanced(team_id, league_id, is_home=True):
    # Récupère les 10 derniers matchs pour pondération
    fixtures = get_api("fixtures", {"team": team_id, "season": SEASON, "last": 10})
    if not fixtures: return 1.3
    
    weighted_goals = 0
    total_weight = 0
    for idx, f in enumerate(reversed(fixtures)):
        weight = 0.9 ** idx
        if f['teams']['home']['id'] == team_id:
            g = f['goals']['home'] or 0
        else:
            g = f['goals']['away'] or 0
        weighted_goals += g * weight
        total_weight += weight
    return weighted_goals / total_weight

# --- INTERFACE PRINCIPALE ---
st.title("ITROZ PREDICTOR ULTIMATE")

tab1, tab2 = st.tabs(["🎯 ANALYSE DÉTAILLÉE (1VS1)", "🚀 SCANNER AUTOMATIQUE"])

# --- ONGLET 1 : ANALYSE 1VS1 (TOUTES LES CATÉGORIES REMISES) ---
with tab1:
    l_name_1v1 = st.selectbox("CHOISIR LA LIGUE", [k for k in LEAGUES_DICT.keys() if k != "🌍 TOUS LES CHAMPIONNATS"])
    l_id_1v1 = LEAGUES_DICT[l_name_1v1]
    
    teams_res = get_api("teams", {"league": l_id_1v1, "season": SEASON})
    teams = {t['team']['name']: t['team']['id'] for t in teams_res}
    
    if teams:
        c1, c2 = st.columns(2)
        t_h = c1.selectbox("DOMICILE", sorted(teams.keys()))
        t_a = c2.selectbox("EXTÉRIEUR", sorted(teams.keys()))
        
        if st.button("LANCER LA PRÉDICTION COMPLÈTE"):
            with st.spinner("Moteur Dixon-Coles + xG en action..."):
                lh = get_team_lambda_advanced(teams[t_h], l_id_1v1, True)
                la = get_team_lambda_advanced(teams[t_a], l_id_1v1, False)
                res = calculate_probs_advanced(lh, la)
                
                # --- MÉTRIQUES ---
                m1, m2, m3 = st.columns(3)
                m1.metric(t_h, f"{res['p_h']*100:.1f}%")
                m2.metric("NUL", f"{res['p_n']*100:.1f}%")
                m3.metric(t_a, f"{res['p_a']*100:.1f}%")

                # --- MODE BET & KELLY (CATÉGORIE V1 REMISE) ---
                st.subheader("🤖 MODE BET")
                st.markdown("<div class='bet-card'>", unsafe_allow_html=True)
                bc1, bc2, bc3, bc4 = st.columns(4)
                br = bc1.number_input("CAPITAL (€)", value=100.0, key="br1")
                ch = bc2.number_input(f"COTE {t_h}", value=2.0)
                cn = bc3.number_input("COTE NUL", value=3.0)
                ca = bc4.number_input(f"COTE {t_a}", value=3.0)
                
                # Calcul meilleur value
                options = [{"n": t_h, "p": res['p_h'], "c": ch}, {"n": "NUL", "p": res['p_n'], "c": cn}, {"n": t_a, "p": res['p_a'], "c": ca}]
                best = max(options, key=lambda x: x['p'] * x['c'])
                if best['p'] * best['c'] > 1.02:
                    k_val = (( (best['c']-1) * best['p'] ) - (1 - best['p'])) / (best['c']-1)
                    st.markdown(f"<div class='verdict-text'>IA RECOMMANDE : {best['n']} | MISE : {max(0, br * k_val * 0.2):.2f}€</div>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

                # --- SCORES PROBABLES (CATÉGORIE V2 REMISE) ---
                st.subheader("🔢 TOP 5 SCORES")
                idx = np.unravel_index(np.argsort(res['matrix'].ravel())[-5:][::-1], res['matrix'].shape)
                sc_cols = st.columns(5)
                for i in range(5):
                    sc_cols[i].markdown(f"**{idx[0][i]} - {idx[1][i]}**\n\n{res['matrix'][idx[0][i], idx[1][i]]*100:.1f}%")

# --- ONGLET 2 : SCANNER (OPTION "TOUS LES CHAMPIONNATS" AJOUTÉE) ---
with tab2:
    st.subheader("🔍 SCANNER DE MARCHÉ")
    col_s1, col_s2, col_s3 = st.columns(3)
    
    l_scan_choice = col_s1.selectbox("CHAMPIONNAT À SCANNER", list(LEAGUES_DICT.keys()))
    scan_date = col_s2.date_input("DATE DU SCAN", datetime.now())
    mode_algo = col_s3.select_slider("TEMPÉRAMENT", options=list(ALGO_MODES.keys()), value="MID")
    
    bankroll_scan = st.number_input("BANKROLL TOTAL (€)", value=100.0, key="br_scan")
    conf = ALGO_MODES[mode_algo]

    if st.button("🚀 LANCER LE SCAN AUTOMATIQUE"):
        all_opps = []
        # Définition des IDs à scanner
        leagues_to_scan = [LEAGUES_DICT[l_scan_choice]] if LEAGUES_DICT[l_scan_choice] != "ALL" else [140, 39, 2, 61, 135, 78]
        
        with st.spinner(f"Scan de {len(leagues_to_scan)} ligue(s) en cours..."):
            for lid in leagues_to_scan:
                fixtures = get_api("fixtures", {"league": lid, "season": SEASON, "date": scan_date.strftime('%Y-%m-%d')})
                for f in fixtures:
                    f_id = f['fixture']['id']
                    # Calcul rapide pour le scan
                    lh = get_team_lambda_advanced(f['teams']['home']['id'], lid)
                    la = get_team_lambda_advanced(f['teams']['away']['id'], lid)
                    probs = calculate_probs_advanced(lh, la)
                    
                    # Cotes
                    odds = get_api("odds", {"fixture": f_id})
                    if odds:
                        for bet in odds[0]['bookmakers'][0]['bets']:
                            if bet['name'] == "Match Winner":
                                for v in bet['values']:
                                    p = probs['p_h'] if v['value']=="Home" else (probs['p_n'] if v['value']=="Draw" else probs['p_a'])
                                    ev = p * float(v['odd'])
                                    if ev >= conf['min_ev']:
                                        all_opps.append({"Match": f"{f['teams']['home']['name']} - {f['teams']['away']['name']}", "Pari": v['value'], "Cote": float(v['odd']), "EV": ev, "P": p})

            # Génération Ticket
            valid = sorted(all_opps, key=lambda x: x['EV'], reverse=True)[:conf['max_legs']]
            if valid:
                cote_t = np.prod([o['Cote'] for o in valid])
                prob_t = np.prod([o['P'] for o in valid])
                mise_f = max(0, bankroll_scan * (((cote_t-1)*prob_t - (1-prob_t))/(cote_t-1)) * conf['kelly'])
                
                st.markdown(f"<div class='verdict-text'>TICKET GÉNÉRÉ : @{cote_t:.2f} | MISE : {mise_f:.2f}€</div>", unsafe_allow_html=True)
                st.table(valid)
            else:
                st.error("Aucune opportunité trouvée.")

st.markdown("<div style='text-align:center; opacity:0.3; margin-top:50px;'>iTrOz Predictor v4.0 - L'ALGO FINAL</div>", unsafe_allow_html=True)
