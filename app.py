import streamlit as st
import requests
import numpy as np
from scipy.stats import poisson
from datetime import datetime, timedelta
import pandas as pd

# --- CONFIGURATION CLEMENTRNXX PREDICTOR V7.0 ---
st.set_page_config(page_title="Clementrnxx Predictor V7.0 - ELITE SCANNER", layout="wide")

DISCORD_WEBHOOK_URL = "https://discord.com/api/webhooks/1453026279275106355/gbYAwBRntm1FCoqoBTz5lj1SCe2ijyeHHYoe4CFYwpzOw2DO-ozcCsgkK_53HhB-kFGE"

st.markdown("""
    <style>
    .stApp { background-image: url("https://media.giphy.com/media/VZrfUvQjXaGEQy1RSn/giphy.gif"); background-size: cover; background-attachment: fixed; }
    .stApp > div:first-child { background-color: rgba(0, 0, 0, 0.93); }
    h1, h2, h3, p, span, label { color: #FFD700 !important; font-family: 'Monospace', sans-serif; }
    div.stButton > button {
        background: linear-gradient(45deg, #FFD700, #BF953F) !important;
        border: none !important; color: black !important;
        border-radius: 10px !important; font-weight: 900; height: 3em; font-size: 1.2rem;
    }
    .status-box { border: 2px solid #FFD700; padding: 20px; border-radius: 15px; background: rgba(0,0,0,0.8); text-align: center; margin-bottom: 20px; }
    </style>
""", unsafe_allow_html=True)

# --- CONFIG API & 6 MODES ---
API_KEY = st.secrets["MY_API_KEY"]
BASE_URL = "https://v3.football.api-sports.io/"
HEADERS = {'x-apisports-key': API_KEY}
SEASON = 2025
LEAGUES_DICT = {"La Liga": 140, "Premier League": 39, "Champions League": 2, "Ligue 1": 61, "Serie A": 135, "Bundesliga": 78}

RISK_LEVELS = {
    "ULTRA-SAFE": {"elite_min": 0.90, "p_min": 0.88, "color": "#00FF00"},
    "SAFE": {"elite_min": 0.75, "p_min": 0.80, "color": "#7FFF00"},
    "MID-SAFE": {"elite_min": 0.65, "p_min": 0.70, "color": "#ADFF2F"},
    "MID": {"elite_min": 0.55, "p_min": 0.60, "color": "#FFD700"},
    "MID-AGGRESSIF": {"elite_min": 0.45, "p_min": 0.50, "color": "#FF8C00"},
    "JACKPOT": {"elite_min": 0.30, "p_min": 0.35, "color": "#FF4500"}
}

# --- FONCTIONS TECHNIQUES ---
@st.cache_data(ttl=3600)
def get_api(endpoint, params):
    try:
        r = requests.get(f"{BASE_URL}{endpoint}", headers=HEADERS, params=params, timeout=12)
        return r.json().get('response', [])
    except: return []

def get_team_stats(team_id, league_id, scope_overall, last_n):
    params = {"team": team_id, "season": SEASON, "last": last_n}
    if not scope_overall: params["league"] = league_id
    f = get_api("fixtures", params)
    if not f: return 1.3, 1.3
    scored = [m['goals']['home'] if m['teams']['home']['id'] == team_id else m['goals']['away'] for m in f if m['goals']['home'] is not None]
    conceded = [m['goals']['away'] if m['teams']['home']['id'] == team_id else m['goals']['home'] for m in f if m['goals']['home'] is not None]
    if not scored: return 1.3, 1.3
    weights = [0.96 ** i for i in range(len(scored))]
    return sum(s * w for s, w in zip(scored, weights)) / sum(weights), sum(c * w for c, w in zip(conceded, weights)) / sum(weights)

def calculate_probs(lh, la):
    matrix = np.zeros((8, 8))
    for x in range(8):
        for y in range(8):
            matrix[x, y] = poisson.pmf(x, lh) * poisson.pmf(y, la)
    matrix /= matrix.sum()
    return {
        "Home": np.sum(np.tril(matrix, -1)), "Draw": np.sum(np.diag(matrix)), "Away": np.sum(np.triu(matrix, 1)),
        "Home/Draw": np.sum(np.tril(matrix, -1)) + np.sum(np.diag(matrix)),
        "Draw/Away": np.sum(np.diag(matrix)) + np.sum(np.triu(matrix, 1)),
        "Yes": np.sum(matrix[1:, 1:]), "No": 1.0 - np.sum(matrix[1:, 1:])
    }

# --- UI INTERFACE ---
st.title("⚡ CLEMENTRNXX ELITE GENERATOR V7.0")

tab1, tab2 = st.tabs(["🎯 ANALYSE 1VS1", "📡 GÉNÉRATEUR DE TICKET FOU"])

with tab1:
    # (Section 1vs1 restaurée comme demandé précédemment)
    st.info("Utilisez l'onglet GÉNÉRATEUR pour le scanner Multi-jours.")

with tab2:
    st.markdown("### 🛠 PARAMÈTRES DU SCANNER")
    c1, c2, c3 = st.columns([2, 2, 1])
    
    selected_leagues = c1.multiselect("SÉLECTION DES LIGUES", list(LEAGUES_DICT.keys()), default=list(LEAGUES_DICT.keys()))
    scope_scan = c2.select_slider("QUALITÉ DES DONNÉES", options=["LEAGUE ONLY", "OVER-ALL"], value="OVER-ALL")
    risk_mode = st.select_slider("MODE DE RISQUE (ALGORITHME ÉLITE)", options=list(RISK_LEVELS.keys()), value="MID")
    
    c4, c5, c6 = st.columns(3)
    max_matches = c4.number_input("LIMITE DE MATCHS DANS LE TICKET", 1, 15, 5)
    last_n_scan = c5.number_input("PROFONDEUR STATS (LAST N)", 5, 50, 15)
    days_to_scan = c6.selectbox("PÉRIODE DE SCAN", ["Aujourd'hui uniquement", "3 Jours consécutifs (FOU)"])

    if st.button("🚀 LANCER LE GÉNÉRATEUR ÉLITE"):
        opps = []
        cfg = RISK_LEVELS[risk_mode]
        
        # Gestion des dates
        scan_dates = [datetime.now()]
        if days_to_scan == "3 Jours consécutifs (FOU)":
            scan_dates.append(datetime.now() + timedelta(days=1))
            scan_dates.append(datetime.now() + timedelta(days=2))

        lids = [LEAGUES_DICT[name] for name in selected_leagues]
        
        with st.spinner("L'algorithme analyse des milliers de points de données..."):
            for date_obj in scan_dates:
                date_str = date_obj.strftime('%Y-%m-%d')
                for lid in lids:
                    fixtures = get_api("fixtures", {"league": lid, "season": SEASON, "date": date_str})
                    for f in fixtures:
                        if f['fixture']['status']['short'] != "NS": continue
                        
                        # Calcul Stats
                        ah, dh = get_team_stats(f['teams']['home']['id'], lid, scope_scan=="OVER-ALL", last_n_scan)
                        aa, da = get_team_stats(f['teams']['away']['id'], lid, scope_scan=="OVER-ALL", last_n_scan)
                        lh, la = (ah * da) ** 0.5 * 1.05, (aa * dh) ** 0.5 * 0.95
                        pr_dict = calculate_probs(lh, la)
                        
                        # Check Cotes
                        odds_data = get_api("odds", {"fixture": f['fixture']['id']})
                        if odds_data and odds_data[0]['bookmakers']:
                            for mkt in odds_data[0]['bookmakers'][0]['bets']:
                                if mkt['name'] in ["Match Winner", "Double Chance", "Both Teams Score"]:
                                    for o in mkt['values']:
                                        p_val = pr_dict.get(o['value'], 0)
                                        cote = float(o['odd'])
                                        
                                        if p_val >= cfg['p_min']:
                                            score_elite = (p_val ** 2) * cote
                                            if score_elite >= cfg['elite_min']:
                                                opps.append({
                                                    "Date": date_str,
                                                    "Match": f"{f['teams']['home']['name']} - {f['teams']['away']['name']}",
                                                    "Pari": f"{mkt['name']}: {o['value']}",
                                                    "Cote": cote,
                                                    "Probabilité": p_val,
                                                    "Score Élite": score_elite
                                                })

        # Filtrage et tri par Score Élite
        final_ticket = sorted(opps, key=lambda x: x['Score Élite'], reverse=True)[:max_matches]
        
        if final_ticket:
            total_cote = np.prod([x['Cote'] for x in final_ticket])
            st.markdown(f"""
                <div class="status-box">
                    <h2 style='color:#FFD700'>TICKET {risk_mode} GÉNÉRÉ</h2>
                    <h1 style='font-size:3.5rem; color:#FFD700'>@{total_cote:.2f}</h1>
                    <p>Nombre de matchs : {len(final_ticket)} | Algorithme : $P^2 \\times Cote$</p>
                </div>
            """, unsafe_allow_html=True)
            
            df_display = pd.DataFrame(final_ticket)
            df_display['Probabilité'] = df_display['Probabilité'].apply(lambda x: f"{x:.1%}")
            st.table(df_display[['Date', 'Match', 'Pari', 'Cote', 'Probabilité', 'Score Élite']])
            
            # Webhook Discord
            discord_content = f"🔥 **NOUVEAU TICKET ÉLITE ({risk_mode})**\n"
            discord_content += f"💰 **COTE TOTALE : @{total_cote:.2f}**\n"
            discord_content += f"📅 Période : {days_to_scan}\n\n"
            for x in final_ticket:
                discord_content += f"✅ {x['Date']} | {x['Match']}\n   👉 **{x['Pari']}** @{x['Cote']} (Score: {x['Score Élite']:.2f})\n\n"
            
            requests.post(DISCORD_WEBHOOK_URL, json={"embeds": [{"title": "CLEMENTRNXX SCANNER V7.0", "description": discord_content, "color": 16766720}]})
            st.success("TICKET ENVOYÉ SUR DISCORD !")
        else:
            st.error("Aucun match n'a passé les filtres Élite. Essayez un mode plus souple.")

st.markdown("""<a href="https://github.com/clementrnx" class="github-link">CLEMENTRNXX - SYSTEM V7.0</a>""", unsafe_allow_html=True)
