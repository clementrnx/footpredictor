import streamlit as st
import requests
import numpy as np
from scipy.stats import poisson
from datetime import datetime

# --- CONFIGURATION ET STYLE ---
st.set_page_config(page_title="iTrOz Predictor - L'ALGO", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #0E1117; }
    h1, h2, h3, p, span, label { color: #FFD700 !important; font-family: 'Monospace', sans-serif; }
    .bet-card {
        background: rgba(255, 255, 255, 0.03);
        padding: 25px; border-radius: 15px;
        border: 1px solid rgba(255, 215, 0, 0.2);
        margin-bottom: 20px;
    }
    .verdict-text {
        font-size: 24px; font-weight: bold; text-align: center; color: #FFD700;
        border: 2px solid #FFD700; padding: 15px; border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# --- CONFIGURATION API ---
API_KEY = st.secrets["MY_API_KEY"]
BASE_URL = "https://v3.football.api-sports.io/"
HEADERS = {'x-apisports-key': API_KEY}
SEASON = 2025
DISCORD_WEBHOOK = "https://discord.com/api/webhooks/1453026279275106355/gbYAwBRntm1FCoqoBTz5lj1SCe2ijyeHHYoe4CFYwpzOw2DO-ozcCsgkK_53HhB-kFGE"

# --- MODES DE L'ALGO ---
ALGO_MODES = {
    "SAFE": {"min_ev": 1.15, "kelly": 0.10, "max_legs": 2, "color": "#00FF7F"},
    "MID SAFE": {"min_ev": 1.10, "kelly": 0.20, "max_legs": 3, "color": "#ADFF2F"},
    "MID": {"min_ev": 1.07, "kelly": 0.35, "max_legs": 4, "color": "#FFD700"},
    "MID AGRESSIF": {"min_ev": 1.04, "kelly": 0.50, "max_legs": 5, "color": "#FF8C00"},
    "AGRESSIF": {"min_ev": 1.02, "kelly": 0.75, "max_legs": 8, "color": "#FF4500"},
    "FOU": {"min_ev": 0.98, "kelly": 1.00, "max_legs": 15, "color": "#FF0000"}
}

# --- FONCTIONS TECHNIQUES ---
@st.cache_data(ttl=3600)
def get_api(endpoint, params):
    try:
        r = requests.get(f"{BASE_URL}{endpoint}", headers=HEADERS, params=params, timeout=10)
        return r.json().get('response', [])
    except: return []

def send_to_discord(ticket_opps, cote, mise, ev, mode_name, color):
    embed = {
        "title": f"🏆 TICKET L'ALGO - MODE {mode_name}",
        "color": int(color.replace("#", ""), 16),
        "fields": [
            {"name": "🏟️ SÉLECTIONS", "value": "\n".join([f"**{o['Match']}**\n└ {o['Pari']} (@{o['Cote']})" for o in ticket_opps]), "inline": False},
            {"name": "📈 COTE", "value": f"**{cote:.2f}**", "inline": True},
            {"name": "💰 MISE", "value": f"**{mise:.2f}€**", "inline": True},
            {"name": "🎯 EV", "value": f"**{ev:.2f}**", "inline": True}
        ],
        "footer": {"text": "iTrOz Predictor • Scan Automatique"},
        "timestamp": datetime.now().isoformat()
    }
    requests.post(DISCORD_WEBHOOK, json={"embeds": [embed]})

# --- CŒUR DE L'ALGO (DIXON-COLES) ---
def calculate_probs(lh, la):
    tau = [-0.13, 0.065, 0.065, 0.13] # Correction Dixon-Coles
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
        "p_btts": np.sum(matrix[1:, 1:]),
        "matrix": matrix
    }

# --- INTERFACE PRINCIPALE ---
st.title("iTrOz Predictor - L'ALGO PRO")

col_l, col_r = st.columns(2)
with col_l:
    leagues = {"La Liga": 140, "Premier League": 39, "Champions League": 2, "Ligue 1": 61, "Serie A": 135}
    l_name = st.selectbox("LIGUE", list(leagues.keys()))
    l_id = leagues[l_name]
with col_r:
    scan_date = st.date_input("DATE DU SCAN", datetime.now())

threshold = st.slider("SEUIL OVER/UNDER", 0.5, 4.5, 2.5, 1.0)
bankroll = st.number_input("BANKROLL TOTAL (€)", value=100.0)
mode_name = st.select_slider("TEMPÉRAMENT DE L'ALGO", options=list(ALGO_MODES.keys()), value="MID")
conf = ALGO_MODES[mode_name]

if st.button("🚀 EXÉCUTER L'ALGO"):
    with st.spinner("Analyse du marché en cours..."):
        fixtures = get_api("fixtures", {"league": l_id, "season": SEASON, "date": scan_date.strftime('%Y-%m-%d')})
        all_opps = []

        for f in fixtures:
            f_id = f['fixture']['id']
            # Simulation Lambda (Simplifiée pour l'exemple, à lier à tes stats xG)
            lh, la = 1.5, 1.2 
            probs = calculate_probs(lh, la)
            
            # Récupération Cotes
            odds_res = get_api("odds", {"fixture": f_id})
            if not odds_res: continue
            
            bookie_bets = odds_res[0]['bookmakers'][0]['bets']
            
            # Mapping des marchés pour L'ALGO
            for bet in bookie_bets:
                if bet['name'] == "Match Winner":
                    for v in bet['values']:
                        p = probs['p_h'] if v['value']=="Home" else (probs['p_n'] if v['value']=="Draw" else probs['p_a'])
                        all_opps.append({"Match": f"{f['teams']['home']['name']} - {f['teams']['away']['name']}", "Pari": v['value'], "Cote": float(v['odd']), "Proba IA": p, "EV": p * float(v['odd'])})
                if bet['name'] == "Both Teams Score" and v['value'] == "Yes":
                    p = probs['p_btts']
                    all_opps.append({"Match": f"{f['teams']['home']['name']} - {f['teams']['away']['name']}", "Pari": "BTTS Oui", "Cote": float(v['odd']), "Proba IA": p, "EV": p * float(v['odd'])})

        # Filtrage et Ticket
        valid_opps = sorted([o for o in all_opps if o['EV'] >= conf['min_ev']], key=lambda x: x['EV'], reverse=True)[:conf['max_legs']]
        
        if valid_opps:
            cote_t = np.prod([o['Cote'] for o in valid_opps])
            prob_t = np.prod([o['Proba IA'] for o in valid_opps])
            ev_t = cote_t * prob_t
            
            # Kelly
            b = cote_t - 1
            k_mise = ((b * prob_t - (1 - prob_t)) / b) * conf['kelly'] if b > 0 else 0
            mise_f = max(0, bankroll * k_mise)

            st.markdown(f"<div class='verdict-text'>TICKET GÉNÉRÉ : {cote_t:.2f} (@{mise_f:.2f}€)</div>", unsafe_allow_html=True)
            st.table(valid_opps)
            
            send_to_discord(valid_opps, cote_t, mise_f, ev_t, mode_name, conf['color'])
            st.toast("Signal transmis à Discord !")
        else:
            st.error("Aucune opportunité détectée par L'ALGO pour ce mode.")

st.markdown("<div style='text-align:center; margin-top:50px; opacity:0.5;'>iTrOz Predictor v3.0 - L'ALGO Propriétaire</div>", unsafe_allow_html=True)
