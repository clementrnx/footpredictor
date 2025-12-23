import streamlit as st
import requests
import numpy as np
from scipy.stats import poisson
from datetime import datetime

# --- INTERFACE ET STYLE PREMIUM ---
st.set_page_config(page_title="iTrOz Predictor V5.0 - PowerValue", layout="wide")

st.markdown("""
    <style>
    .stApp {
        background-image: url("https://media.giphy.com/media/VZrfUvQjXaGEQy1RSn/giphy.gif");
        background-size: cover;
        background-attachment: fixed;
    }
    .stApp > div:first-child { background-color: rgba(0, 0, 0, 0.95); }
    h1, h2, h3, p, span, label { color: #FFD700 !important; font-family: 'Segoe UI', sans-serif; }
    
    .info-card {
        background: rgba(255, 215, 0, 0.03); border: 1px solid rgba(255, 215, 0, 0.2);
        padding: 20px; border-radius: 15px; margin-bottom: 15px;
    }
    .metric-box {
        text-align: center; padding: 15px; border-radius: 10px;
        background: rgba(255, 255, 255, 0.05); border: 1px solid #FFD700;
    }
    .stButton>button {
        background: linear-gradient(45deg, #FFD700, #B8860B) !important;
        color: black !important; font-weight: bold; border: none !important;
        letter-spacing: 2px; transition: 0.3s;
    }
    .stButton>button:hover { transform: scale(1.02); box-shadow: 0 0 15px #FFD700; }
    </style>
""", unsafe_allow_html=True)

# --- CONFIGURATION API ---
API_KEY = st.secrets["MY_API_KEY"]
BASE_URL = "https://v3.football.api-sports.io/"
HEADERS = {'x-apisports-key': API_KEY}
SEASON = 2025

LEAGUES_DICT = {
    "🌍 TOUS LES CHAMPIONNATS": "ALL",
    "🇪🇸 La Liga": 140, "🇬🇧 Premier League": 39, "🇪🇺 Champions League": 2, 
    "🇫🇷 Ligue 1": 61, "🇮🇹 Serie A": 135, "🇩🇪 Bundesliga": 78, "🇳🇱 Eredivisie": 88
}

# --- FONCTIONS TECHNIQUES ---
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
    return {"p_h": np.sum(np.tril(matrix, -1)), "p_n": np.sum(np.diag(matrix)), "p_a": np.sum(np.triu(matrix, 1)), "matrix": matrix}

def get_lambda(team_id, league_id):
    f = get_api("fixtures", {"team": team_id, "season": SEASON, "last": 12})
    if not f: return 1.25
    goals = [(m['goals']['home'] if m['teams']['home']['id'] == team_id else m['goals']['away']) or 0 for m in f]
    weights = [0.92**i for i in range(len(goals))]
    return sum(g * w for g, w in zip(reversed(goals), weights)) / sum(weights)

# --- APPLICATION ---
st.title("🔱 ITROZ PREDICTOR V5.0 - POWER-VALUE")

t_1v1, t_scan = st.tabs(["🎯 ANALYSE DÉTAILLÉE 1VS1", "⚡ SCANNER HAUT RENDEMENT"])

with t_1v1:
    l_box = st.selectbox("CHAMPIONNAT", [k for k in LEAGUES_DICT.keys() if k != "🌍 TOUS LES CHAMPIONNATS"])
    teams_res = get_api("teams", {"league": LEAGUES_DICT[l_box], "season": SEASON})
    teams = {t['team']['name']: t['team']['id'] for t in teams_res}
    
    if teams:
        c1, c2 = st.columns(2)
        home_t, away_t = c1.selectbox("DOMICILE", sorted(teams.keys())), c2.selectbox("EXTÉRIEUR", sorted(teams.keys()))
        
        if st.button("LANCER L'AUDIT COMPLET"):
            lh, la = get_lambda(teams[home_t], LEAGUES_DICT[l_box]), get_lambda(teams[away_t], LEAGUES_DICT[l_box])
            st.session_state.v5 = {"res": calculate_probs(lh, la), "h": home_t, "a": away_t}

    if 'v5' in st.session_state:
        res, h, a = st.session_state.v5["res"], st.session_state.v5["h"], st.session_state.v5["a"]
        
        # --- SECTION PROBABILITÉS ---
        st.subheader("📊 PROBABILITÉS IA")
        sc1, sc2, sc3 = st.columns(3)
        sc1.markdown(f"<div class='metric-box'><b>{h}</b><br><span style='font-size:25px'>{res['p_h']*100:.1f}%</span></div>", unsafe_allow_html=True)
        sc2.markdown(f"<div class='metric-box'><b>NUL</b><br><span style='font-size:25px'>{res['p_n']*100:.1f}%</span></div>", unsafe_allow_html=True)
        sc3.markdown(f"<div class='metric-box'><b>{a}</b><br><span style='font-size:25px'>{res['p_a']*100:.1f}%</span></div>", unsafe_allow_html=True)

        # --- SECTION AUDIT (DISTINCTE) ---
        st.markdown("---")
        st.subheader("🕵️ AUDIT DE VOTRE PARI")
        with st.container():
            ac1, ac2 = st.columns(2)
            user_bet = ac1.selectbox("VOTRE SÉLECTION", [h, "Nul", a, f"{h}/Nul", f"Nul/{a}"])
            user_odd = ac2.number_input("COTE DU BOOKMAKER", value=1.50)
            
            p_user = res['p_h'] if user_bet == h else (res['p_n'] if user_bet == "Nul" else res['p_a'])
            if "/" in user_bet: p_user = (res['p_h']+res['p_n']) if h in user_bet else (res['p_a']+res['p_n'])
            
            ev = p_user * user_odd
            st.markdown(f"<div class='info-card' style='text-align:center'>INDICE DE FIABILITÉ : <b>{ev:.2f}</b><br>Verdict : {'✅ VALIDE' if ev > 1.05 else '⚠️ RISQUÉ'}</div>", unsafe_allow_html=True)

        # --- SECTION MODE BET (CONSEILS) ---
        st.subheader("💰 CONSEILS DE MISE (MODE BET)")
        with st.container():
            st.markdown("<div class='info-card'>", unsafe_allow_html=True)
            bc1, bc2 = st.columns(2)
            bankroll = bc1.number_input("VOTRE CAPITAL (€)", value=100.0)
            # Recommandation basée sur le critère de Kelly fractionné
            b_val = (user_odd - 1)
            mise_k = max(0, ((b_val * p_user) - (1 - p_user)) / b_val) if b_val > 0 else 0
            st.write(f"Conseil IA : Mise de **{(bankroll * mise_k * 0.25):.2f} €** (Gestion prudente 25% Kelly)")
            st.markdown("</div>", unsafe_allow_html=True)

with t_scan:
    st.subheader("⚡ TICKET POWER-VALUE (GAINS OPTIMISÉS / RISQUE MINIME)")
    st.write("L'IA cherche les meilleures opportunités combinées de la journée.")
    
    sc_col1, sc_col2 = st.columns(2)
    l_scan = sc_col1.selectbox("LIGUE", list(LEAGUES_DICT.keys()), key="lscan")
    d_scan = sc_col2.date_input("DATE", datetime.now(), key="dscan")
    
    if st.button("GÉNÉRER LE MEILLEUR TICKET"):
        lids = [LEAGUES_DICT[l_scan]] if LEAGUES_DICT[l_scan] != "ALL" else [140, 39, 2, 61, 135, 78, 88]
        all_opps = []
        
        with st.spinner("Calcul du ratio Gain/Risque..."):
            for lid in lids:
                fixtures = get_api("fixtures", {"league": lid, "season": SEASON, "date": d_scan.strftime('%Y-%m-%d')})
                for f in fixtures:
                    lh, la = get_lambda(f['teams']['home']['id'], lid), get_lambda(f['teams']['away']['id'], lid)
                    pr = calculate_probs(lh, la)
                    
                    # On scanne Domicile, Nul, Extérieur
                    for choice, p, val in [("Home", pr['p_h'], "Home"), ("Draw", pr['p_n'], "Draw"), ("Away", pr['p_a'], "Away")]:
                        if p > 0.60: # Sécurité de base : minimum 60% de chance
                            odds_data = get_api("odds", {"fixture": f['fixture']['id']})
                            if odds_data:
                                for b in odds_data[0]['bookmakers'][0]['bets']:
                                    if b['name'] == "Match Winner":
                                        for v in b['values']:
                                            if v['value'] == val:
                                                cote = float(v['odd'])
                                                if cote >= 1.15: # On cherche quand même un peu de gain
                                                    # SCORE POWER-VALUE = Proba * Cote (plus c'est haut, meilleur est le ratio)
                                                    all_opps.append({
                                                        "Match": f"{f['teams']['home']['name']} - {f['teams']['away']['name']}",
                                                        "Pari": f"{v['value']}",
                                                        "Probabilité": f"{p*100:.1f}%",
                                                        "Cote": cote,
                                                        "Power-Value": p * cote
                                                    })

            # Sélection des 3 meilleurs Power-Value (Équilibre parfait risque/gain)
            top_ticket = sorted(all_opps, key=lambda x: x['Power-Value'], reverse=True)[:3]
            
            if top_ticket:
                cote_f = np.prod([x['Cote'] for x in top_ticket])
                st.markdown(f"<div class='verdict-box'><h2 style='color:white'>TICKET POWER-VALUE GÉNÉRÉ</h2>"
                            f"<p style='font-size:20px'>COTE TOTALE : @{cote_f:.2f} | PROBABILITÉ COMBINÉE : {np.prod([float(x['Probabilité'].replace('%',''))/100 for x in top_ticket])*100:.1f}%</p></div>", unsafe_allow_html=True)
                st.table(top_ticket)
            else:
                st.error("Aucune opportunité à haut rendement détectée pour cette date.")

st.markdown("<div style='text-align:center; opacity:0.3; margin-top:50px;'>ITROZ PREDICTOR V5.0 - THE GOLD STANDARD</div>", unsafe_allow_html=True)
