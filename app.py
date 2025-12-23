import streamlit as st
import requests
import numpy as np
from scipy.stats import poisson
from datetime import datetime, timedelta
import pandas as pd

# --- CONFIGURATION CLEMENTRNXX PREDICTOR V9.0 ---
st.set_page_config(page_title="Clementrnxx Predictor V9.0 - TOTAL CONTROL", layout="wide")

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
    .verdict-box { border: 2px dotted #FFD700; padding: 20px; border-radius: 15px; background: rgba(0,0,0,0.8); margin: 20px 0; }
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
        "Home": np.sum(np.tril(matrix, -1)), "Draw": np.sum(np.diag(matrix)), "Away": np.sum(np.triu(matrix, 1)),
        "Home/Draw": np.sum(np.tril(matrix, -1)) + np.sum(np.diag(matrix)),
        "Draw/Away": np.sum(np.diag(matrix)) + np.sum(np.triu(matrix, 1)),
        "Home/Away": np.sum(np.tril(matrix, -1)) + np.sum(np.triu(matrix, 1)),
        "Yes": np.sum(matrix[1:, 1:]), "No": 1.0 - np.sum(matrix[1:, 1:])
    }

# --- UI ---
tab1, tab2 = st.tabs(["🎯 AUDIT 1VS1 (FULL)", "📡 GÉNÉRATEUR MULTI-MARCHÉS"])

with tab1:
    st.subheader("🕵️ AUDIT APPROFONDI DU MATCH")
    c1, c2, c3 = st.columns([2, 2, 1])
    l_name = c1.selectbox("LIGUE", list(LEAGUES_DICT.keys()), key="1v1_l")
    scope_1v1 = c2.select_slider("SCOPE DATA", options=["LEAGUE ONLY", "OVER-ALL"], value="OVER-ALL", key="1v1_sc")
    last_n_1v1 = c3.number_input("MATCHS (LAST N)", 5, 50, 15, key="1v1_n")
    
    teams = {t['team']['name']: t['team']['id'] for t in get_api("teams", {"league": LEAGUES_DICT[l_name], "season": SEASON})}
    
    if teams:
        col1, col2 = st.columns(2)
        th, ta = col1.selectbox("DOMICILE", sorted(teams.keys())), col2.selectbox("EXTÉRIEUR", sorted(teams.keys()))
        if st.button("LANCER L'AUDIT COMPLET"):
            ah, dh = get_team_stats(teams[th], LEAGUES_DICT[l_name], scope_1v1=="OVER-ALL", last_n_1v1)
            aa, da = get_team_stats(teams[ta], LEAGUES_DICT[l_name], scope_1v1=="OVER-ALL", last_n_1v1)
            lh, la = (ah * da) ** 0.5 * 1.05, (aa * dh) ** 0.5 * 0.95
            st.session_state.v9_res = {"res": calculate_perfect_probs(lh, la), "th": th, "ta": ta}

    if 'v9_res' in st.session_state:
        r, th, ta = st.session_state.v9_res["res"], st.session_state.v9_res["th"], st.session_state.v9_res["ta"]
        st.markdown(f"### 📈 Probabilités Auditées : {th} vs {ta}")
        m = st.columns(4)
        m[0].metric(f"Victoire {th}", f"{r['Home']:.1%}"); m[1].metric("Match Nul", f"{r['Draw']:.1%}"); m[2].metric(f"Victoire {ta}", f"{r['Away']:.1%}"); m[3].metric("BTTS OUI", f"{r['Yes']:.1%}")
        
        st.markdown("<div class='verdict-box'>", unsafe_allow_html=True)
        st.subheader("💰 CALCULATEUR DE VALUE & BETTING")
        v1, v2, v3 = st.columns(3); ch = v1.number_input(f"Cote {th}", 1.0); cn = v2.number_input("Cote N", 1.0); ca = v3.number_input(f"Cote {ta}", 1.0)
        v4, v5, v6 = st.columns(3); c1n = v4.number_input("Cote 1N", 1.0); cn2 = v5.number_input("Cote N2", 1.0); c12 = v6.number_input("Cote 12", 1.0)
        v7, v8 = st.columns(2); cby = v7.number_input("BTTS OUI", 1.0); cbn = v8.number_input("BTTS NON", 1.0)
        
        all_bets = [
            (f"Victoire {th}", ch, r['Home']), ("Match Nul", cn, r['Draw']), (f"Victoire {ta}", ca, r['Away']),
            ("Double Chance 1N", c1n, r['Home/Draw']), ("Double Chance N2", cn2, r['Draw/Away']), 
            ("Double Chance 12", c12, r['Home/Away']), ("BTTS OUI", cby, r['Yes']), ("BTTS NON", cbn, r['No'])
        ]
        
        for name, cote, prob in all_bets:
            if cote > 1.0:
                ev, score = (prob * cote), (prob**2 * cote)
                if ev > 1.05:
                    st.success(f"🔥 VALUE : {name} @{cote:.2f} | EV: {ev:.2f} | Score Élite: {score:.2f}")
        st.markdown("</div>", unsafe_allow_html=True)

with tab2:
    st.subheader("📡 SCANNER ÉLITE TOUS MARCHÉS")
    d1, d2, d3 = st.columns([2, 2, 2])
    start_d, end_d = d1.date_input("DÉBUT", datetime.now()), d2.date_input("FIN", datetime.now() + timedelta(days=2))
    l_scan = d3.selectbox("LIGUES", ["TOUTES"] + list(LEAGUES_DICT.keys()), key="gen_l")

    g1, g2, g3 = st.columns(3)
    risk_mode = g1.select_slider("NIVEAU DE RISQUE", options=list(RISK_LEVELS.keys()), value="MID")
    max_m = g2.number_input("MAX MATCHS", 1, 15, 5)
    scope_g = g3.select_slider("SCOPE SCANNER", options=["LEAGUE ONLY", "OVER-ALL"], value="OVER-ALL", key="gen_sc")

    if st.button("🔥 GÉNÉRER LE TICKET ÉLITE"):
        opps = []
        cfg = RISK_LEVELS[risk_mode]
        lids = LEAGUES_DICT.values() if l_scan == "TOUTES" else [LEAGUES_DICT[l_scan]]
        dates = [start_d + timedelta(days=i) for i in range((end_d - start_d).days + 1)]
        
        pb = st.progress(0)
        for i, d in enumerate(dates):
            d_s = d.strftime('%Y-%m-%d')
            for lid in lids:
                fixtures = get_api("fixtures", {"league": lid, "season": SEASON, "date": d_s})
                for f in fixtures:
                    if f['fixture']['status']['short'] != "NS": continue
                    ah, dh = get_team_stats(f['teams']['home']['id'], lid, scope_g=="OVER-ALL", 15)
                    aa, da = get_team_stats(f['teams']['away']['id'], lid, scope_g=="OVER-ALL", 15)
                    lh, la = (ah * da) ** 0.5 * 1.05, (aa * dh) ** 0.5 * 0.95
                    pr = calculate_perfect_probs(lh, la)
                    
                    odds = get_api("odds", {"fixture": f['fixture']['id']})
                    if odds and odds[0]['bookmakers']:
                        for mkt in odds[0]['bookmakers'][0]['bets']:
                            if mkt['name'] in ["Match Winner", "Double Chance", "Both Teams Score"]:
                                for o in mkt['values']:
                                    # Mapping dynamique des marchés API vers notre dict de probas
                                    val_map = {"Home": "Home", "Draw": "Draw", "Away": "Away", "Home/Draw": "Home/Draw", "Draw/Away": "Draw/Away", "Home/Away": "Home/Away", "Yes": "Yes", "No": "No"}
                                    p_val = pr.get(val_map.get(o['value'], ""), 0)
                                    
                                    if p_val >= cfg['p_min']:
                                        cote = float(o['odd'])
                                        score = (p_val**2) * cote
                                        if score >= cfg['elite_min']:
                                            opps.append({"Date": d_s, "Match": f"{f['teams']['home']['name']}-{f['teams']['away']['name']}", "Pari": f"{mkt['name']}: {o['value']}", "Cote": cote, "Score": score})
            pb.progress((i + 1) / len(dates))

        final = sorted(opps, key=lambda x: x['Score'], reverse=True)[:max_m]
        if final:
            total_c = np.prod([x['Cote'] for x in final])
            st.table(pd.DataFrame(final))
            st.metric("COTE TOTALE", f"@{total_c:.2f}")
            
            # Discord
            msg = f"🏆 **TICKET ÉLITE V9 ({risk_mode})**\n📅 {start_d} au {end_d}\n\n"
            msg += "\n".join([f"🔹 {x['Date']} | {x['Match']}\n   👉 {x['Pari']} @{x['Cote']}" for x in final])
            msg += f"\n\n🔥 **COTE TOTALE : @{total_c:.2f}**"
            requests.post(DISCORD_WEBHOOK_URL, json={"embeds": [{"title": "CLEMENTRNXX PREDICTOR V9", "description": msg, "color": 16766720}]})
            st.success("TICKET ENVOYÉ !")
