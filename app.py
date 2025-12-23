import streamlit as st
import requests
import numpy as np
from scipy.stats import poisson
from datetime import datetime

# --- CONFIGURATION ET STYLE AURA GOLD ---
st.set_page_config(page_title="L'ALGO • iTrOz", layout="wide")

# Remplace par ton URL de GIF
GIF_URL = "https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExNHJueGZ3bmZ3bmZ3bmZ3bmZ3bmZ3bmZ3bmZ3bmZ3bmZ3bmZ3JmVwPXYxX2ludGVybmFsX2dpZl9ieV9pZCZjdD1n/3o7TKMGpxx6U9R6T6M/giphy.gif"

st.markdown(f"""
    <style>
    .stApp {{
        background: linear-gradient(rgba(0, 0, 0, 0.9), rgba(0, 0, 0, 0.9)), url("{GIF_URL}");
        background-size: cover; background-attachment: fixed;
    }}
    h1, h2, h3, p, span, label, div {{ color: #FFD700 !important; font-family: 'Monospace', sans-serif; }}
    .bet-card {{
        background: rgba(0, 0, 0, 0.8); padding: 25px; border-radius: 15px;
        border: 2px solid #FFD700; margin-bottom: 20px;
    }}
    .stButton>button {{
        width: 100%; background-color: #FFD700 !important; color: black !important;
        font-weight: bold; border: none; height: 50px;
    }}
    .stNumberInput input, .stSelectbox div, .stDateInput input, .stSlider div {{
        background-color: black !important; color: #FFD700 !important; border: 1px solid #FFD700 !important;
    }}
    table {{ background-color: rgba(0,0,0,0.8) !important; color: #FFD700 !important; border: 1px solid #FFD700; }}
    </style>
""", unsafe_allow_html=True)

# --- CONFIGURATION API & MODES ---
API_KEY = st.secrets["MY_API_KEY"]
BASE_URL = "https://v3.football.api-sports.io/"
HEADERS = {'x-apisports-key': API_KEY}
DISCORD_WEBHOOK = "https://discord.com/api/webhooks/1453026279275106355/gbYAwBRntm1FCoqoBTz5lj1SCe2ijyeHHYoe4CFYwpzOw2DO-ozcCsgkK_53HhB-kFGE"

ALGO_MODES = {
    "SAFE": {"min_ev": 1.15, "kelly": 0.10, "max_legs": 2, "color": "#FFD700"},
    "MID SAFE": {"min_ev": 1.10, "kelly": 0.20, "max_legs": 3, "color": "#FFD700"},
    "MID": {"min_ev": 1.07, "kelly": 0.35, "max_legs": 4, "color": "#FFD700"},
    "MID AGRESSIF": {"min_ev": 1.04, "kelly": 0.50, "max_legs": 5, "color": "#FFD700"},
    "AGRESSIF": {"min_ev": 1.02, "kelly": 0.75, "max_legs": 8, "color": "#FFD700"},
    "FOU": {"min_ev": 0.98, "kelly": 1.00, "max_legs": 15, "color": "#FF4B4B"}
}

# --- FONCTION MATHÉMATIQUE (DIXON-COLES) ---
def get_dc_probs(lh, la):
    tau = [-0.13, 0.065, 0.065, 0.13]
    matrix = np.zeros((10, 10))
    for x in range(10):
        for y in range(10):
            p = poisson.pmf(x, lh) * poisson.pmf(y, la)
            if x==0 and y==0: p *= (1 + tau[0]*lh*la)
            elif x==1 and y==0: p *= (1 + tau[1]*lh)
            elif x==0 and y==1: p *= (1 + tau[2]*la)
            elif x==1 and y==1: p *= (1 + tau[3])
            matrix[x, y] = max(p, 0)
    matrix /= matrix.sum()
    return matrix

# --- INTERFACE ---
st.markdown("<h1 style='text-align:center;'>L'ALGO v3.0 PRO</h1>", unsafe_allow_html=True)

with st.sidebar:
    st.header("⚙️ PARAMÈTRES")
    bankroll = st.number_input("BANKROLL TOTAL (€)", value=100.0)
    mode_name = st.select_slider("TEMPÉRAMENT", options=list(ALGO_MODES.keys()), value="MID")
    threshold = st.slider("SEUIL OVER/UNDER", 0.5, 4.5, 2.5, 1.0)
    scan_date = st.date_input("DATE", datetime.now())

conf = ALGO_MODES[mode_name]

if st.button("LANCER LE SCAN DES MARCHÉS"):
    with st.spinner("L'ALGO analyse les données API Pro..."):
        # 1. On récupère les matchs de la ligue (Ex: Premier League ID 39)
        fixtures = requests.get(f"{BASE_URL}fixtures", headers=HEADERS, params={"league": 39, "season": 2025, "date": scan_date.strftime('%Y-%m-%d')}).json().get('response', [])
        
        all_opportunities = []

        for f in fixtures:
            f_id = f['fixture']['id']
            # On simule ici les lambda (lh, la) via tes stats xG habituelles
            lh, la = 1.6, 1.2 
            matrix = get_dc_probs(lh, la)
            
            # Calcul des probabilités par catégorie
            p_h = np.sum(np.tril(matrix, -1))
            p_n = np.sum(np.diag(matrix))
            p_a = np.sum(np.triu(matrix, 1))
            p_1n, p_n2, p_12 = p_h + p_n, p_n + p_a, p_h + p_a
            p_btts = np.sum(matrix[1:, 1:])
            p_over = np.sum([matrix[x,y] for x in range(10) for y in range(10) if x+y > threshold])

            # Récupération des cotes réelles
            odds_res = requests.get(f"{BASE_URL}odds", headers=HEADERS, params={"fixture": f_id}).json().get('response', [])
            if not odds_res: continue
            
            # Extraction des cotes et calcul de l'EV (Expected Value)
            for bet in odds_res[0]['bookmakers'][0]['bets']:
                match_name = f"{f['teams']['home']['name']} - {f['teams']['away']['name']}"
                
                # Catégorie 1N2
                if bet['name'] == "Match Winner":
                    for v in bet['values']:
                        p = p_h if v['value']=="Home" else (p_n if v['value']=="Draw" else p_a)
                        all_opportunities.append({"Match": match_name, "Pari": v['value'], "Cote": float(v['odd']), "Proba": p, "EV": p * float(v['odd'])})
                
                # Catégorie BTTS
                if bet['name'] == "Both Teams Score":
                    for v in bet['values']:
                        if v['value'] == "Yes":
                            all_opportunities.append({"Match": match_name, "Pari": "BTTS OUI", "Cote": float(v['odd']), "Proba": p_btts, "EV": p_btts * float(v['odd'])})

                # Catégorie Over/Under
                if bet['name'] == "Goals Over/Under":
                    for v in bet['values']:
                        if v['value'] == f"Over {threshold}":
                            all_opportunities.append({"Match": match_name, "Pari": f"Over {threshold}", "Cote": float(v['odd']), "Proba": p_over, "EV": p_over * float(v['odd'])})

        # --- FILTRAGE ET TICKET ---
        valid_opps = sorted([o for o in all_opportunities if o['EV'] >= conf['min_ev']], key=lambda x: x['EV'], reverse=True)[:conf['max_legs']]

        if valid_opps:
            cote_t = np.prod([o['Cote'] for o in valid_opps])
            prob_t = np.prod([o['Proba'] for o in valid_opps])
            ev_t = cote_t * prob_t
            
            # Kelly Criterion
            b = cote_t - 1
            k_mise = ((b * prob_t - (1 - prob_t)) / b) * conf['kelly'] if b > 0 else 0
            mise_f = max(0, bankroll * k_mise)

            # Affichage UI
            st.markdown(f"""
                <div class="bet-card">
                    <h2 style="text-align:center;">🎫 TICKET GÉNÉRÉ : MODE {mode_name}</h2>
                    <p style="text-align:center; font-size:25px;">Cote Totale : <b>{cote_t:.2f}</b> | Mise : <b>{mise_f:.2f}€</b></p>
                </div>
            """, unsafe_allow_html=True)
            st.table(valid_opps)

            # Envoi Webhook Discord
            embed = {
                "title": f"🔱 SIGNAL L'ALGO - {mode_name}",
                "color": 16766464,
                "fields": [
                    {"name": "📑 MATCHS", "value": "\n".join([f"• {o['Match']} : `{o['Pari']}` (@{o['Cote']})" for o in valid_opps])},
                    {"name": "📊 STATS", "value": f"Cote: **{cote_t:.2f}** | Mise: **{mise_f:.2f}€** | EV: **{ev_t:.2f}**"}
                ]
            }
            requests.post(DISCORD_WEBHOOK, json={"embeds": [embed]})
            st.success("SIGNAL ENVOYÉ SUR DISCORD")
        else:
            st.warning("L'ALGO n'a trouvé aucune opportunité rentable.")
