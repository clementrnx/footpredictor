import streamlit as st
import requests
import numpy as np
from scipy.stats import poisson
from datetime import datetime
import pandas as pd

# --- CONFIGURATION CLEMENTRNXX PREDICTOR V6.0 ---
st.set_page_config(page_title="Clementrnxx Predictor V6.0", layout="wide")

DISCORD_WEBHOOK_URL = "https://discord.com/api/webhooks/1453026279275106355/gbYAwBRntm1FCoqoBTz5lj1SCe2ijyeHHYoe4CFYwpzOw2DO-ozcCsgkK_53HhB-kFGE"

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
    .verdict-box { border: 2px solid #FFD700; padding: 20px; border-radius: 15px; background: rgba(0,0,0,0.8); margin-bottom: 20px; }
    </style>
""", unsafe_allow_html=True)

# --- API CONFIG ---
API_KEY = st.secrets["MY_API_KEY"]
BASE_URL = "https://v3.football.api-sports.io/"
HEADERS = {'x-apisports-key': API_KEY}
SEASON = 2025
LEAGUES_DICT = {"La Liga": 140, "Premier League": 39, "Champions League": 2, "Ligue 1": 61, "Serie A": 135, "Bundesliga": 78}

RISK_LEVELS = {
    "ULTRA-SAFE": {"elite_min": 0.85, "p_min": 0.85},
    "SAFE": {"elite_min": 0.70, "p_min": 0.75},
    "MID-SAFE": {"elite_min": 0.60, "p_min": 0.65},
    "MID": {"elite_min": 0.50, "p_min": 0.55},
    "MID-AGGRESSIF": {"elite_min": 0.40, "p_min": 0.45},
    "JACKPOT": {"elite_min": 0.25, "p_min": 0.35}
}

# --- FUNCTIONS ---
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
        "p_12": np.sum(np.tril(matrix, -1)) + np.sum(np.triu(matrix, 1)),
        "p_btts": np.sum(matrix[1:, 1:]), "p_nobtts": 1.0 - np.sum(matrix[1:, 1:])
    }

# --- UI ---
tab1, tab2, tab3 = st.tabs(["🎯 ANALYSE 1VS1", "📡 SCANNER ÉLITE", "📊 CLASSEMENTS"])

with tab1:
    st.subheader("🛠 CONFIGURATION DE L'ANALYSE")
    c_l, c_s, c_n = st.columns([2, 2, 1])
    l_name = c_l.selectbox("LIGUE", list(LEAGUES_DICT.keys()), key="1v1_l")
    scope_1v1 = c_s.select_slider("SCOPE DATA", options=["LEAGUE ONLY", "OVER-ALL"], value="OVER-ALL")
    last_n_1v1 = c_n.number_input("MATCHS", 5, 50, 15)
    
    teams_res = get_api("teams", {"league": LEAGUES_DICT[l_name], "season": SEASON})
    teams = {t['team']['name']: t['team']['id'] for t in teams_res}
    
    if teams:
        col1, col2 = st.columns(2)
        th = col1.selectbox("DOMICILE", sorted(teams.keys()))
        ta = col2.selectbox("EXTÉRIEUR", sorted(teams.keys()))
        
        if st.button("LANCER L'ANALYSE PROFONDE"):
            ah, dh = get_team_stats(teams[th], LEAGUES_DICT[l_name], scope_1v1=="OVER-ALL", last_n_1v1)
            aa, da = get_team_stats(teams[ta], LEAGUES_DICT[l_name], scope_1v1=="OVER-ALL", last_n_1v1)
            lh, la = (ah * da) ** 0.5 * 1.05, (aa * dh) ** 0.5 * 0.95
            st.session_state.v6_res = {"res": calculate_perfect_probs(lh, la), "th": th, "ta": ta}

    if 'v6_res' in st.session_state:
        r, th, ta = st.session_state.v6_res["res"], st.session_state.v6_res["th"], st.session_state.v6_res["ta"]
        
        st.markdown(f"### 📊 Probabilités : {th} vs {ta}")
        m = st.columns(5)
        m[0].metric(th, f"{r['p_h']:.1%}"); m[1].metric("NUL", f"{r['p_n']:.1%}"); m[2].metric(ta, f"{r['p_a']:.1%}")
        m[3].metric("BTTS OUI", f"{r['p_btts']:.1%}"); m[4].metric("BTTS NON", f"{r['p_nobtts']:.1%}")
        
        st.markdown("---")
        st.subheader("💰 CALCULATEUR DE VALUE (TOUTES CATÉGORIES)")
        
        v1, v2, v3, v4 = st.columns(4)
        c_h = v1.number_input(f"Cote {th}", 1.0); c_n = v2.number_input("Cote NUL", 1.0)
        c_a = v3.number_input(f"Cote {ta}", 1.0); c_1n = v4.number_input(f"{th}/N", 1.0)
        
        v5, v6, v7, v8 = st.columns(4)
        c_n2 = v5.number_input(f"N/{ta}", 1.0); c_12 = v6.number_input("12 (Sans Nul)", 1.0)
        c_by = v7.number_input("BTTS OUI", 1.0); c_bn = v8.number_input("BTTS NON", 1.0)
        
        bets = [
            (f"Victoire {th}", c_h, r['p_h']), ("Match Nul", c_n, r['p_n']), (f"Victoire {ta}", c_a, r['p_a']),
            ("Double Chance 1N", c_1n, r['p_1n']), ("Double Chance N2", c_n2, r['p_n2']), ("Double Chance 12", c_12, r['p_12']),
            ("BTTS OUI", c_by, r['p_btts']), ("BTTS NON", c_bn, r['p_nobtts'])
        ]
        
        st.markdown("### 🎯 VERDICT VALUE")
        for name, cote, prob in bets:
            if cote > 1.0:
                ev = cote * prob
                score_elite = (prob**2) * cote
                if ev > 1.05:
                    st.success(f"🔥 **{name}** | EV: {ev:.2f} | Score Élite: {score_elite:.2f}")

with tab2:
    st.subheader("📡 SCANNER DE TICKETS AUTOMATISÉ")
    sc1, sc2, sc3 = st.columns([2, 2, 1])
    l_scan = sc1.selectbox("LIGUES", ["TOUTES"] + list(LEAGUES_DICT.keys()), key="sc_l")
    scope_scan = sc2.select_slider("SCOPE DATA (SCANNER)", options=["LEAGUE ONLY", "OVER-ALL"], value="OVER-ALL", key="sc_s")
    risk_mode = st.select_slider("PHILOSOPHIE DE JEU", options=list(RISK_LEVELS.keys()), value="MID")
    
    sc4, sc5, sc6 = st.columns(3)
    max_matches = sc4.number_input("LIMITE DE NOMBRE DE MATCHS", 1, 20, 5)
    last_n_scan = sc5.number_input("PROFONDEUR DATA (LAST N)", 5, 50, 15, key="sc_n")
    scan_date = sc6.date_input("DATE DU MATCH", datetime.now())

    if st.button("🔥 GÉNÉRER TICKET ÉLITE & DISCORD"):
        # (Logique du scanner identique avec envoi Webhook...)
        lids = LEAGUES_DICT.values() if l_scan == "TOUTES" else [LEAGUES_DICT[l_scan]]
        opps = []
        cfg = RISK_LEVELS[risk_mode]
        for lid in lids:
            fixtures = get_api("fixtures", {"league": lid, "season": SEASON, "date": scan_date.strftime('%Y-%m-%d')})
            for f in fixtures:
                if f['fixture']['status']['short'] != "NS": continue
                ah, dh = get_team_stats(f['teams']['home']['id'], lid, scope_scan=="OVER-ALL", last_n_scan)
                aa, da = get_team_stats(f['teams']['away']['id'], lid, scope_scan=="OVER-ALL", last_n_scan)
                lh, la = (ah * da) ** 0.5 * 1.05, (aa * dh) ** 0.5 * 0.95
                pr = calculate_perfect_probs(lh, la)
                
                odds = get_api("odds", {"fixture": f['fixture']['id']})
                if odds and odds[0]['bookmakers']:
                    for mkt in odds[0]['bookmakers'][0]['bets']:
                        if mkt['name'] in ["Match Winner", "Double Chance", "Both Teams Score"]:
                            for o in mkt['values']:
                                cote = float(o['odd'])
                                p_val = 0
                                if o['value'] == 'Home': p_val = pr['p_h']
                                elif o['value'] == 'Draw': p_val = pr['p_n']
                                elif o['value'] == 'Away': p_val = pr['p_a']
                                elif o['value'] == 'Home/Draw': p_val = pr['p_1n']
                                elif o['value'] == 'Draw/Away': p_val = pr['p_n2']
                                elif o['value'] == 'Yes': p_val = pr['p_btts']
                                
                                if p_val >= cfg['p_min']:
                                    score = (p_val**2) * cote
                                    if score >= cfg['elite_min']:
                                        opps.append({"M": f"{f['teams']['home']['name']}-{f['teams']['away']['name']}", "P": o['value'], "C": cote, "Pr": p_val, "S": score})

        final = sorted(opps, key=lambda x: x['S'], reverse=True)[:max_matches]
        if final:
            st.table(pd.DataFrame(final))
            t_msg = f"🏆 **TICKET {risk_mode}**\n*Limite: {max_matches} matchs*\n\n" + "\n".join([f"🔹 {x['M']} : {x['P']} @{x['C']}" for x in final])
            requests.post(DISCORD_WEBHOOK_URL, json={"embeds": [{"title": "CLEMENTRNXX PREDICTOR", "description": t_msg, "color": 16766720}]})
            st.success("✅ Envoyé sur Discord !")

with tab3:
    st.subheader("📊 CLASSEMENTS")
    l_sel = st.selectbox("LIGUE", list(LEAGUES_DICT.keys()), key="st_l")
    standings = get_api("standings", {"league": LEAGUES_DICT[l_sel], "season": SEASON})
    if standings:
        df = pd.DataFrame([{"Equipe": t['team']['name'], "Pts": t['points'], "Forme": t['form']} for t in standings[0]['league']['standings'][0]])
        st.dataframe(df, use_container_width=True)
