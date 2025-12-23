import streamlit as st
import requests
import numpy as np
from scipy.stats import poisson
from datetime import datetime, timedelta
import pandas as pd

# --- CONFIGURATION CLEMENTRNXX PREDICTOR V8.0 ---
st.set_page_config(page_title="Clementrnxx Predictor V8.0", layout="wide")

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

# --- CONFIG API & 6 MODES ---
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
        "p_12": np.sum(np.tril(matrix, -1)) + np.sum(np.triu(matrix, 1)),
        "p_btts": np.sum(matrix[1:, 1:]), "p_nobtts": 1.0 - np.sum(matrix[1:, 1:])
    }

# --- UI ---
tab1, tab2, tab3 = st.tabs(["🎯 ANALYSE 1VS1 COMPLETE", "📡 GÉNÉRATEUR ELITE 3-JOURS", "📊 CLASSEMENTS"])

with tab1:
    st.subheader("🛠 ANALYSE PERSONNALISÉE")
    c_l, c_s, c_n = st.columns([2, 2, 1])
    l_name = c_l.selectbox("LIGUE", list(LEAGUES_DICT.keys()), key="1v1_l")
    scope_1v1 = c_s.select_slider("SCOPE DATA", options=["LEAGUE ONLY", "OVER-ALL"], value="OVER-ALL")
    last_n_1v1 = c_n.number_input("PROFONDEUR (N)", 5, 50, 15)
    
    teams_res = get_api("teams", {"league": LEAGUES_DICT[l_name], "season": SEASON})
    teams = {t['team']['name']: t['team']['id'] for t in teams_res}
    
    if teams:
        col1, col2 = st.columns(2)
        th, ta = col1.selectbox("DOMICILE", sorted(teams.keys())), col2.selectbox("EXTÉRIEUR", sorted(teams.keys()))
        
        if st.button("LANCER L'ANALYSE"):
            ah, dh = get_team_stats(teams[th], LEAGUES_DICT[l_name], scope_1v1=="OVER-ALL", last_n_1v1)
            aa, da = get_team_stats(teams[ta], LEAGUES_DICT[l_name], scope_1v1=="OVER-ALL", last_n_1v1)
            lh, la = (ah * da) ** 0.5 * 1.05, (aa * dh) ** 0.5 * 0.95
            st.session_state.v8_1v1 = {"res": calculate_perfect_probs(lh, la), "th": th, "ta": ta}

    if 'v8_1v1' in st.session_state:
        r, th, ta = st.session_state.v8_1v1["res"], st.session_state.v8_1v1["th"], st.session_state.v8_1v1["ta"]
        st.markdown(f"#### 📈 Probabilités : {th} vs {ta}")
        m = st.columns(5)
        m[0].metric("HOME", f"{r['p_h']:.1%}"); m[1].metric("NUL", f"{r['p_n']:.1%}"); m[2].metric("AWAY", f"{r['p_a']:.1%}")
        m[3].metric("BTTS OUI", f"{r['p_btts']:.1%}"); m[4].metric("BTTS NON", f"{r['p_nobtts']:.1%}")
        
        st.markdown("#### 💰 CALCULATEUR DE VALUE")
        v_c1, v_c2, v_c3 = st.columns(3); ch = v_c1.number_input(f"Cote {th}", 1.0); cn = v_c2.number_input("Cote NUL", 1.0); ca = v_c3.number_input(f"Cote {ta}", 1.0)
        v_c4, v_c5, v_c6 = st.columns(3); c1n = v_c4.number_input("Cote 1N", 1.0); cn2 = v_c5.number_input("Cote N2", 1.0); c12 = v_c6.number_input("Cote 12", 1.0)
        v_c7, v_c8 = st.columns(2); c_by = v_c7.number_input("BTTS OUI", 1.0); c_bn = v_c8.number_input("BTTS NON", 1.0)
        
        bets = [(f"Victoire {th}", ch, r['p_h']), ("Match Nul", cn, r['p_n']), (f"Victoire {ta}", ca, r['p_a']), ("Double Chance 1N", c1n, r['p_1n']), ("Double Chance N2", cn2, r['p_n2']), ("BTTS OUI", c_by, r['p_btts'])]
        for name, cote, prob in bets:
            if cote > 1.0 and (cote * prob) > 1.05:
                st.success(f"🔥 VALUE : {name} @{cote} (Prob: {prob:.1%}) | Score: {(prob**2)*cote:.2f}")

with tab2:
    st.subheader("📡 GÉNÉRATEUR DE TICKET MULTI-JOURS")
    g1, g2, g3 = st.columns([2, 2, 2])
    l_scan = g1.selectbox("LIGUES", ["TOUTES LES LIGUES"] + list(LEAGUES_DICT.keys()))
    scope_scan = g2.select_slider("QUALITÉ DATA", options=["LEAGUE ONLY", "OVER-ALL"], value="OVER-ALL", key="g_sc")
    start_date = g3.date_input("DÉBUTER LE SCAN LE :", datetime.now())
    
    g4, g5, g6 = st.columns(3)
    risk_mode = g4.select_slider("RISQUE", options=list(RISK_LEVELS.keys()), value="MID")
    max_matches = g5.number_input("NOMBRE DE MATCHS MAX", 1, 15, 5)
    last_n_scan = g6.number_input("PROFONDEUR (LAST N)", 5, 50, 15, key="g_n")

    if st.button("🚀 GÉNÉRER LE TICKET SUR 3 JOURS"):
        opps = []
        cfg = RISK_LEVELS[risk_mode]
        lids = LEAGUES_DICT.values() if l_scan == "TOUTES LES LIGUES" else [LEAGUES_DICT[l_scan]]
        
        # Scan de 3 jours consécutifs à partir de start_date
        dates_to_scan = [start_date + timedelta(days=i) for i in range(3)]
        
        for d in dates_to_scan:
            d_str = d.strftime('%Y-%m-%d')
            for lid in lids:
                fixtures = get_api("fixtures", {"league": lid, "season": SEASON, "date": d_str})
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
                                    p_val = 0
                                    if o['value'] == 'Home': p_val = pr['p_h']
                                    elif o['value'] == 'Draw': p_val = pr['p_n']
                                    elif o['value'] == 'Away': p_val = pr['p_a']
                                    elif o['value'] == 'Home/Draw': p_val = pr['p_1n']
                                    elif o['value'] == 'Draw/Away': p_val = pr['p_n2']
                                    elif o['value'] == 'Yes': p_val = pr['p_btts']
                                    
                                    if p_val >= cfg['p_min']:
                                        score = (p_val**2) * float(o['odd'])
                                        if score >= cfg['elite_min']:
                                            opps.append({"Date": d_str, "Match": f"{f['teams']['home']['name']}-{f['teams']['away']['name']}", "Pari": o['value'], "Cote": float(o['odd']), "Score": score})

        final = sorted(opps, key=lambda x: x['Score'], reverse=True)[:max_matches]
        if final:
            total_cote = np.prod([x['Cote'] for x in final])
            st.table(pd.DataFrame(final))
            st.metric("COTE TOTALE", f"@{total_cote:.2f}")
            
            # Envoi Discord
            msg = f"🏆 **TICKET ELITE - MODE {risk_mode}**\n📅 Scan du {start_date} au {dates_to_scan[-1]}\n\n"
            msg += "\n".join([f"🔹 {x['Date']} | {x['Match']} : **{x['Pari']}** @{x['Cote']}" for x in final])
            msg += f"\n\n🔥 **COTE TOTALE : @{total_cote:.2f}**"
            requests.post(DISCORD_WEBHOOK_URL, json={"embeds": [{"title": "CLEMENTRNXX PREDICTOR", "description": msg, "color": 16766720}]})
            st.success("TICKET ENVOYÉ !")

with tab3:
    st.subheader("📊 CLASSEMENTS")
    l_sel = st.selectbox("LIGUE", list(LEAGUES_DICT.keys()), key="st_l")
    standings = get_api("standings", {"league": LEAGUES_DICT[l_sel], "season": SEASON})
    if standings:
        df = pd.DataFrame([{"Pos": t['rank'], "Equipe": t['team']['name'], "Pts": t['points'], "Forme": t['form']} for t in standings[0]['league']['standings'][0]])
        st.dataframe(df, use_container_width=True)
