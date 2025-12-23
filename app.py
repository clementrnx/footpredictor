import streamlit as st
import requests
import numpy as np
from scipy.stats import poisson
from datetime import datetime, timedelta
import pandas as pd

# --- CONFIGURATION CLEMENTRNXX PREDICTOR V8.5 ---
st.set_page_config(page_title="Clementrnxx Predictor V8.5 - DATE RANGE", layout="wide")

DISCORD_WEBHOOK_URL = "https://discord.com/api/webhooks/1453026279275106355/gbYAwBRntm1FCoqoBTz5lj1SCe2ijyeHHYoe4CFYwpzOw2DO-ozcCsgkK_53HhB-kFGE"

st.markdown("""
    <style>
    .stApp { background-image: url("https://media.giphy.com/media/VZrfUvQjXaGEQy1RSn/giphy.gif"); background-size: cover; background-attachment: fixed; }
    .stApp > div:first-child { background-color: rgba(0, 0, 0, 0.93); }
    h1, h2, h3, p, span, label { color: #FFD700 !important; font-family: 'Monospace', sans-serif; }
    div.stButton > button {
        background: linear-gradient(45deg, #FFD700, #BF953F) !important;
        border: none !important; color: black !important;
        border-radius: 10px !important; font-weight: 900; height: 3em; width: 100%;
    }
    .verdict-box { border: 2px solid #FFD700; padding: 20px; border-radius: 15px; background: rgba(0,0,0,0.8); margin-bottom: 20px; }
    </style>
""", unsafe_allow_html=True)

# --- CONFIG API & MODES ---
API_KEY = st.secrets["MY_API_KEY"]
BASE_URL = "https://v3.football.api-sports.io/"
HEADERS = {'x-apisports-key': API_KEY}
SEASON = 2025
LEAGUES_DICT = {"La Liga": 140, "Premier League": 39, "Champions League": 2, "Ligue 1": 61, "Serie A": 135, "Bundesliga": 78}

RISK_LEVELS = {
    "ULTRA-SAFE": {"elite_min": 0.90, "p_min": 0.88},
    "SAFE": {"elite_min": 0.75, "p_min": 0.80},
    "MID-SAFE": {"elite_min": 0.65, "p_min": 0.70},
    "MID": {"elite_min": 0.55, "p_min": 0.60},
    "MID-AGGRESSIF": {"elite_min": 0.45, "p_min": 0.50},
    "JACKPOT": {"elite_min": 0.30, "p_min": 0.35}
}

# --- FONCTIONS ---
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
        "p_btts": np.sum(matrix[1:, 1:]), "p_nobtts": 1.0 - np.sum(matrix[1:, 1:])
    }

# --- UI ---
tab1, tab2 = st.tabs(["🎯 ANALYSE 1VS1 COMPLETE", "📡 GÉNÉRATEUR DE TICKET (RANGE)"])

with tab1:
    # (Mode 1vs1 complet avec toutes les options de cotes)
    st.subheader("🛠 ANALYSE 1VS1")
    c_l, c_s, c_n = st.columns([2, 2, 1])
    l_name = c_l.selectbox("LIGUE", list(LEAGUES_DICT.keys()), key="1v1_l")
    scope_1v1 = c_s.select_slider("SCOPE DATA", options=["LEAGUE ONLY", "OVER-ALL"], value="OVER-ALL", key="1v1_sc")
    last_n_1v1 = c_n.number_input("DERNIERS MATCHS", 5, 50, 15, key="1v1_n")
    
    teams_res = get_api("teams", {"league": LEAGUES_DICT[l_name], "season": SEASON})
    teams = {t['team']['name']: t['team']['id'] for t in teams_res}
    
    if teams:
        col1, col2 = st.columns(2)
        th, ta = col1.selectbox("DOMICILE", sorted(teams.keys())), col2.selectbox("EXTÉRIEUR", sorted(teams.keys()))
        if st.button("LANCER L'ANALYSE"):
            ah, dh = get_team_stats(teams[th], LEAGUES_DICT[l_name], scope_1v1=="OVER-ALL", last_n_1v1)
            aa, da = get_team_stats(teams[ta], LEAGUES_DICT[l_name], scope_1v1=="OVER-ALL", last_n_1v1)
            lh, la = (ah * da) ** 0.5 * 1.05, (aa * dh) ** 0.5 * 0.95
            st.session_state.v85_res = {"res": calculate_perfect_probs(lh, la), "th": th, "ta": ta}

    if 'v85_res' in st.session_state:
        r, th, ta = st.session_state.v85_res["res"], st.session_state.v85_res["th"], st.session_state.v85_res["ta"]
        st.markdown(f"#### 📈 Probabilités : {th} vs {ta}")
        st.columns(5)[0].metric("HOME", f"{r['p_h']:.1%}") # ... (Reste des probas)

with tab2:
    st.subheader("📡 SCANNER MULTI-DATES")
    
    # Nouvelle zone de sélection de dates
    d1, d2, d3 = st.columns([2, 2, 2])
    start_date = d1.date_input("DU :", datetime.now())
    end_date = d2.date_input("AU :", datetime.now() + timedelta(days=3))
    l_scan = d3.selectbox("LIGUES", ["TOUTES LES LIGUES"] + list(LEAGUES_DICT.keys()))

    g1, g2, g3 = st.columns(3)
    risk_mode = g1.select_slider("MODE DE RISQUE", options=list(RISK_LEVELS.keys()), value="MID")
    max_matches = g2.number_input("MAX MATCHS DANS LE TICKET", 1, 15, 5)
    scope_scan = g3.select_slider("SCOPE SCANNER", options=["LEAGUE ONLY", "OVER-ALL"], value="OVER-ALL")

    if st.button("🔥 GÉNÉRER LE COMBINÉ SUR LA PÉRIODE"):
        opps = []
        cfg = RISK_LEVELS[risk_mode]
        lids = LEAGUES_DICT.values() if l_scan == "TOUTES LES LIGUES" else [LEAGUES_DICT[l_scan]]
        
        # Calcul de la liste des dates entre Début et Fin
        delta = end_date - start_date
        dates_list = [start_date + timedelta(days=i) for i in range(delta.days + 1)]
        
        if len(dates_list) > 10:
            st.warning("⚠️ Plage de dates trop large (max 10 jours). Réduction automatique.")
            dates_list = dates_list[:10]

        progress_bar = st.progress(0)
        for idx, d in enumerate(dates_list):
            d_str = d.strftime('%Y-%m-%d')
            for lid in lids:
                fixtures = get_api("fixtures", {"league": lid, "season": SEASON, "date": d_str})
                for f in fixtures:
                    if f['fixture']['status']['short'] != "NS": continue
                    ah, dh = get_team_stats(f['teams']['home']['id'], lid, scope_scan=="OVER-ALL", 15)
                    aa, da = get_team_stats(f['teams']['away']['id'], lid, scope_scan=="OVER-ALL", 15)
                    lh, la = (ah * da) ** 0.5 * 1.05, (aa * dh) ** 0.5 * 0.95
                    pr = calculate_perfect_probs(lh, la)
                    
                    odds = get_api("odds", {"fixture": f['fixture']['id']})
                    if odds and odds[0]['bookmakers']:
                        for mkt in odds[0]['bookmakers'][0]['bets']:
                            if mkt['name'] in ["Match Winner", "Double Chance", "Both Teams Score"]:
                                for o in mkt['values']:
                                    p_val = 0
                                    if o['value'] == 'Home': p_val = pr['p_h']
                                    elif o['value'] == 'Draw': p_val = pr['p_n']
                                    elif o['value'] == 'Away': p_val = pr['p_a']
                                    elif o['value'] == 'Home/Draw': p_val = pr['p_1n']
                                    elif o['value'] == 'Draw/Away': p_val = pr['p_n2']
                                    elif o['value'] == 'Yes': p_val = pr['p_btts']
                                    
                                    if p_val >= cfg['p_min']:
                                        cote = float(o['odd'])
                                        score = (p_val**2) * cote
                                        if score >= cfg['elite_min']:
                                            opps.append({
                                                "Date": d_str,
                                                "Match": f"{f['teams']['home']['name']} - {f['teams']['away']['name']}",
                                                "Pari": o['value'],
                                                "Cote": cote,
                                                "Score": score
                                            })
            progress_bar.progress((idx + 1) / len(dates_list))

        final = sorted(opps, key=lambda x: x['Score'], reverse=True)[:max_matches]
        if final:
            total_cote = np.prod([x['Cote'] for x in final])
            st.table(pd.DataFrame(final))
            st.metric("COTE TOTALE DU COMBINÉ", f"@{total_cote:.2f}")
            
            # Webhook Discord
            msg = f"🚀 **COMBINÉ SUR MESURE ({risk_mode})**\n📅 Période : du {start_date} au {end_date}\n\n"
            msg += "\n".join([f"✅ {x['Date']} | {x['Match']} : **{x['Pari']}** @{x['Cote']}" for x in final])
            msg += f"\n\n🔥 **COTE TOTALE : @{total_cote:.2f}**"
            requests.post(DISCORD_WEBHOOK_URL, json={"embeds": [{"title": "CLEMENTRNXX ELITE RANGE SCANNER", "description": msg, "color": 16766720}]})
            st.success("TICKET ENVOYÉ SUR DISCORD !")
        else:
            st.error("Aucune opportunité trouvée sur cette période avec ces réglages.")
