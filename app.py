import streamlit as st
import requests
import numpy as np
from scipy.stats import poisson
from datetime import datetime
import pandas as pd

# --- CONFIGURATION CLEMENTRNXX PREDICTOR V5.6 ---
st.set_page_config(page_title="Clementrnxx Predictor V5.6", layout="wide")

DISCORD_WEBHOOK_URL = st.secrets.get("DISCORD_WEBHOOK", "")

st.markdown("""
    <style>
    .stApp { background-image: url("https://media.giphy.com/media/VZrfUvQjXaGEQy1RSn/giphy.gif"); background-size: cover; background-attachment: fixed; }
    .stApp > div:first-child { background-color: rgba(0, 0, 0, 0.93); }
    h1, h2, h3, p, span, label { color: #FFD700 !important; font-family: 'Monospace', sans-serif; }
    div.stButton > button {
        background: rgba(255, 215, 0, 0.1) !important; backdrop-filter: blur(10px);
        border: 2px solid #FFD700 !important; color: #FFD700 !important;
        border-radius: 15px !important; font-weight: 900; transition: 0.4s; width: 100%;
    }
    </style>
""", unsafe_allow_html=True)

# --- CONFIG API ---
API_KEY = st.secrets["MY_API_KEY"]
BASE_URL = "https://v3.football.api-sports.io/"
HEADERS = {'x-apisports-key': API_KEY}
SEASON = 2025
LEAGUES_DICT = {"La Liga": 140, "Premier League": 39, "Champions League": 2, "Ligue 1": 61, "Serie A": 135, "Bundesliga": 78}

RISK_LEVELS = {
    "SAFE": {"elite_min": 0.75, "p_min": 0.80},
    "MID": {"elite_min": 0.55, "p_min": 0.60},
    "AGGRESSIF": {"elite_min": 0.35, "p_min": 0.40}
}

# --- LOGIQUE CORE ---
@st.cache_data(ttl=3600)
def get_api(endpoint, params):
    try:
        r = requests.get(f"{BASE_URL}{endpoint}", headers=HEADERS, params=params, timeout=12)
        return r.json().get('response', [])
    except: return []

def get_team_stats(team_id, league_id, scope_overall, last_n=15):
    params = {"team": team_id, "season": SEASON, "last": last_n}
    if not scope_overall: params["league"] = league_id
    f = get_api("fixtures", params)
    if not f: return 1.3, 1.3
    scored = [m['goals']['home'] if m['teams']['home']['id'] == team_id else m['goals']['away'] for m in f if m['goals']['home'] is not None]
    conceded = [m['goals']['away'] if m['teams']['home']['id'] == team_id else m['goals']['home'] for m in f if m['goals']['home'] is not None]
    if not scored: return 1.3, 1.3
    weights = [0.96 ** i for i in range(len(scored))]
    return sum(s * w for s, w in zip(scored, weights)) / sum(weights), sum(c * w for c, w in zip(conceded, weights)) / sum(weights)

def calculate_perfect_probs(lh, la):
    matrix = np.zeros((10, 10))
    for x in range(10):
        for y in range(10):
            matrix[x, y] = poisson.pmf(x, lh) * poisson.pmf(y, la)
    matrix /= matrix.sum()
    return {
        "p_h": np.sum(np.tril(matrix, -1)), "p_n": np.sum(np.diag(matrix)), "p_a": np.sum(np.triu(matrix, 1)),
        "p_1n": np.sum(np.tril(matrix, -1)) + np.sum(np.diag(matrix)),
        "p_n2": np.sum(np.diag(matrix)) + np.sum(np.triu(matrix, 1)),
        "p_btts": np.sum(matrix[1:, 1:]), "matrix": matrix
    }

# --- UI ---
st.title("⚽ CLEMENTRNXX PREDICTOR V5.6")

tab1, tab2 = st.tabs(["🎯 ANALYSE 1VS1", "📡 SCANNER ÉLITE"])

with tab1:
    col_l, col_s, col_n = st.columns([2, 2, 1])
    l_name = col_l.selectbox("LIGUE", list(LEAGUES_DICT.keys()))
    scope_1v1 = col_s.select_slider("SCOPE DATA (1vs1)", options=["LEAGUE ONLY", "OVER-ALL"], value="OVER-ALL")
    last_n_1v1 = col_n.number_input("MATCHS", 5, 50, 15, key="n1v1")
    # ... (Reste du code 1vs1 identique à la V5.5)

with tab2:
    st.subheader("📡 SCANNER DE TICKETS AUTOMATISÉ")
    sc1, sc2, sc3 = st.columns([2, 2, 1])
    l_scan = sc1.selectbox("LIGUES", ["TOUTES"] + list(LEAGUES_DICT.keys()))
    scope_scan = sc2.select_slider("SCOPE DATA (SCANNER)", options=["LEAGUE ONLY", "OVER-ALL"], value="OVER-ALL")
    risk_mode = sc3.selectbox("PHILOSOPHIE", list(RISK_LEVELS.keys()), index=1)
    
    scan_date = st.date_input("DATE DU SCAN", datetime.now())

    if st.button("🔥 GÉNÉRER TICKET ÉLITE & DISCORD"):
        lids = LEAGUES_DICT.values() if l_scan == "TOUTES" else [LEAGUES_DICT[l_scan]]
        opps = []
        progress = st.progress(0)
        
        for idx, lid in enumerate(lids):
            fixtures = get_api("fixtures", {"league": lid, "season": SEASON, "date": scan_date.strftime('%Y-%m-%d')})
            for f in fixtures:
                if f['fixture']['status']['short'] != "NS": continue
                # Application du Scope sélectionné ici
                is_overall = (scope_scan == "OVER-ALL")
                ah, dh = get_team_stats(f['teams']['home']['id'], lid, is_overall)
                aa, da = get_team_stats(f['teams']['away']['id'], lid, is_overall)
                
                lh, la = (ah * da) ** 0.5 * 1.05, (aa * dh) ** 0.5 * 0.95
                pr = calculate_perfect_probs(lh, la)
                
                odds = get_api("odds", {"fixture": f['fixture']['id']})
                if odds and odds[0]['bookmakers']:
                    for mkt in odds[0]['bookmakers'][0]['bets']:
                        if mkt['name'] in ["Match Winner", "Double Chance"]:
                            for o in mkt['values']:
                                cote = float(o['odd'])
                                p_val = 0
                                if o['value'] == 'Home': p_val = pr['p_h']
                                elif o['value'] == 'Draw': p_val = pr['p_n']
                                elif o['value'] == 'Away': p_val = pr['p_a']
                                elif o['value'] == 'Home/Draw': p_val = pr['p_1n']
                                
                                if p_val >= RISK_LEVELS[risk_mode]['p_min']:
                                    score = (p_val**2) * cote
                                    if score >= RISK_LEVELS[risk_mode]['elite_min']:
                                        opps.append({"Match": f"{f['teams']['home']['name']} vs {f['teams']['away']['name']}", "Pronostic": o['value'], "Cote": cote, "Prob": f"{p_val:.1%}", "Score": score})
            progress.progress((idx + 1) / len(lids))

        final = sorted(opps, key=lambda x: x['Score'], reverse=True)
        if final:
            st.table(pd.DataFrame(final).head(10))
            
            # Webhook Discord
            t_msg = f"📊 **NOUVEAU TICKET ÉLITE ({risk_mode})**\n*Scope: {scope_scan}*\n\n"
            t_msg += "\n".join([f"🔹 {x['Match']} : **{x['Pronostic']}** @{x['Cote']} ({x['Prob']})" for x in final[:3]])
            
            if DISCORD_WEBHOOK_URL:
                requests.post(DISCORD_WEBHOOK_URL, json={"embeds": [{"title": "CLEMENTRNXX SCANNER", "description": t_msg, "color": 16766720}]})
                st.success("✅ Ticket envoyé sur Discord !")
        else:
            st.warning("Aucune opportunité Élite trouvée pour cette configuration.")
