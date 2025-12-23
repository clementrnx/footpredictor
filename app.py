import streamlit as st
import requests
import numpy as np
from scipy.stats import poisson
from datetime import datetime
import pandas as pd
import time

# --- CONFIGURATION CLEMENTRNXX PREDICTOR V5.5 ---
st.set_page_config(page_title="Clementrnxx Predictor V5.5", layout="wide")

# URL de ton Webhook Discord (à configurer dans tes secrets ou ici)
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
    .verdict-box { border: 2px solid #FFD700; padding: 25px; text-align: center; border-radius: 20px; background: rgba(0,0,0,0.8); margin: 20px 0; }
    .github-link { display: block; text-align: center; color: #FFD700 !important; font-weight: bold; text-decoration: none; margin-top: 40px; }
    </style>
""", unsafe_allow_html=True)

# --- CONFIG API & PHILOSOPHIES ---
API_KEY = st.secrets["MY_API_KEY"]
BASE_URL = "https://v3.football.api-sports.io/"
HEADERS = {'x-apisports-key': API_KEY}
SEASON = 2025
LEAGUES_DICT = {"La Liga": 140, "Premier League": 39, "Champions League": 2, "Ligue 1": 61, "Serie A": 135, "Bundesliga": 78}

RISK_LEVELS = {
    "SAFE": {"elite_min": 0.75, "p_min": 0.80, "kelly": 0.04},
    "MID": {"elite_min": 0.55, "p_min": 0.60, "kelly": 0.08},
    "AGGRESSIF": {"elite_min": 0.35, "p_min": 0.40, "kelly": 0.15}
}

# --- FONCTIONS ---
def send_discord_webhook(title, content, color=16766720):
    if not DISCORD_WEBHOOK_URL: return
    data = {"embeds": [{"title": title, "description": content, "color": color}]}
    requests.post(DISCORD_WEBHOOK_URL, json=data)

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
    matrix = np.zeros((12, 12))
    for x in range(12):
        for y in range(12):
            matrix[x, y] = poisson.pmf(x, lh) * poisson.pmf(y, la)
    matrix /= matrix.sum()
    return {
        "p_h": np.sum(np.tril(matrix, -1)), "p_n": np.sum(np.diag(matrix)), "p_a": np.sum(np.triu(matrix, 1)),
        "p_1n": np.sum(np.tril(matrix, -1)) + np.sum(np.diag(matrix)),
        "p_n2": np.sum(np.diag(matrix)) + np.sum(np.triu(matrix, 1)),
        "p_12": np.sum(np.tril(matrix, -1)) + np.sum(np.triu(matrix, 1)),
        "p_btts": np.sum(matrix[1:, 1:]), "p_nobtts": 1.0 - np.sum(matrix[1:, 1:]), "matrix": matrix
    }

# --- UI ---
st.title("⚽ CLEMENTRNXX PREDICTOR V5.5")

tab1, tab2, tab3 = st.tabs(["🎯 ANALYSE 1VS1", "📡 SCANNER ÉLITE", "📊 STATS"])

with tab1:
    col_l, col_s, col_n = st.columns([2, 2, 1])
    l_name = col_l.selectbox("LIGUE", list(LEAGUES_DICT.keys()), key="1v1_l")
    scope_1v1 = col_s.select_slider("SCOPE DATA", options=["LEAGUE ONLY", "OVER-ALL"], value="OVER-ALL")
    last_n = col_n.number_input("DERNIERS MATCHS", 5, 50, 15)
    
    teams_res = get_api("teams", {"league": LEAGUES_DICT[l_name], "season": SEASON})
    teams = {t['team']['name']: t['team']['id'] for t in teams_res}
    
    if teams:
        c1, c2 = st.columns(2)
        th, ta = c1.selectbox("DOMICILE", sorted(teams.keys())), c2.selectbox("EXTÉRIEUR", sorted(teams.keys()))
        
        if st.button("LANCER L'ANALYSE"):
            ah, dh = get_team_stats(teams[th], LEAGUES_DICT[l_name], scope_1v1=="OVER-ALL", last_n)
            aa, da = get_team_stats(teams[ta], LEAGUES_DICT[l_name], scope_1v1=="OVER-ALL", last_n)
            lh, la = (ah * da) ** 0.5 * 1.05, (aa * dh) ** 0.5 * 0.95
            st.session_state.v5_final = {"res": calculate_perfect_probs(lh, la), "th": th, "ta": ta}

    if 'v5_final' in st.session_state:
        r = st.session_state.v5_final["res"]
        st.markdown(f"### 📈 Probabilités pour {st.session_state.v5_final['th']} vs {st.session_state.v5_final['ta']}")
        m = st.columns(5)
        m[0].metric("HOME", f"{r['p_h']:.1%}"); m[1].metric("DRAW", f"{r['p_n']:.1%}"); m[2].metric("AWAY", f"{r['p_a']:.1%}")
        m[3].metric("BTTS OUI", f"{r['p_btts']:.1%}"); m[4].metric("BTTS NON", f"{r['p_nobtts']:.1%}")

        st.subheader("💰 CALCULATEUR DE VALUE")
        ic = st.columns(4)
        c_h = ic[0].number_input("Cote Home", 1.0); c_n = ic[1].number_input("Cote Nul", 1.0)
        c_a = ic[2].number_input("Cote Away", 1.0); c_1n = ic[3].number_input("Cote 1N", 1.0)
        ic2 = st.columns(4)
        c_n2 = ic2[0].number_input("Cote N2", 1.0); c_12 = ic2[1].number_input("Cote 12", 1.0)
        c_by = ic2[2].number_input("Cote BTTS OUI", 1.0); c_bn = ic2[3].number_input("Cote BTTS NON", 1.0)

        bets = [("Home", c_h, r['p_h']), ("Nul", c_n, r['p_n']), ("Away", c_a, r['p_a']), ("1N", c_1n, r['p_1n']), 
                ("N2", c_n2, r['p_n2']), ("12", c_12, r['p_12']), ("BTTS Y", c_by, r['p_btts']), ("BTTS N", c_bn, r['p_nobtts'])]
        
        for name, cote, prob in bets:
            if cote > 1.0 and (prob * cote) > 1.05:
                st.success(f"🔥 VALUE DÉTECTÉE : {name} | EV: {(prob*cote):.2f} | Score Élite: {(prob**2 * cote):.2f}")

with tab2:
    st.subheader("📡 SCANNER DE TICKETS AUTOMATISÉ")
    sc1, sc2, sc3 = st.columns([2, 1, 1])
    l_scan = sc1.selectbox("LIGUES", ["TOUTES"] + list(LEAGUES_DICT.keys()), key="sc_l")
    d_scan = sc2.date_input("DATE", datetime.now())
    risk_mode = sc3.select_slider("PHILOSOPHIE", options=list(RISK_LEVELS.keys()), value="MID")
    
    if st.button("GÉNÉRER LE MEILLEUR TICKET & ENVOYER DISCORD"):
        lids = LEAGUES_DICT.values() if l_scan == "TOUTES" else [LEAGUES_DICT[l_scan]]
        opps = []
        with st.spinner("Sniping en cours..."):
            for lid in lids:
                fixtures = get_api("fixtures", {"league": lid, "season": SEASON, "date": d_scan.strftime('%Y-%m-%d')})
                for f in fixtures:
                    if f['fixture']['status']['short'] != "NS": continue
                    ah, dh = get_team_stats(f['teams']['home']['id'], lid, True)
                    aa, da = get_team_stats(f['teams']['away']['id'], lid, True)
                    lh, la = (ah * da) ** 0.5 * 1.05, (aa * dh) ** 0.5 * 0.95
                    pr = calculate_perfect_probs(lh, la)
                    
                    odds = get_api("odds", {"fixture": f['fixture']['id']})
                    if odds and odds[0]['bookmakers']:
                        for market in odds[0]['bookmakers'][0]['bets']:
                            if market['name'] in ["Match Winner", "Double Chance", "Both Teams Score"]:
                                for o in market['values']:
                                    p_val = 0
                                    # Mapping (Simplifié pour la démo)
                                    val = o['value']
                                    if val == 'Home': p_val = pr['p_h']
                                    elif val == 'Draw': p_val = pr['p_n']
                                    elif val == 'Away': p_val = pr['p_a']
                                    elif val == 'Home/Draw': p_val = pr['p_1n']
                                    elif val == 'Yes': p_val = pr['p_btts']
                                    
                                    if p_val >= RISK_LEVELS[risk_mode]['p_min']:
                                        cote = float(o['odd'])
                                        score = (p_val**2) * cote
                                        if score >= RISK_LEVELS[risk_mode]['elite_min']:
                                            opps.append({"M": f"{f['teams']['home']['name']}-{f['teams']['away']['name']}", "P": val, "C": cote, "Pr": p_val, "S": score})

        final = sorted(opps, key=lambda x: x['S'], reverse=True)
        if final:
            st.table(pd.DataFrame(final).head(10))
            ticket_msg = "\n".join([f"✅ {x['M']} | {x['P']} @{x['C']} ({x['Pr']:.0%})" for x in final[:5]])
            send_discord_webhook(f"🚀 TICKET ÉLITE - {risk_mode}", f"Cote Totale estimée : @{np.prod([x['C'] for x in final[:3]]):.2f}\n\n{ticket_msg}")
            st.info("Ticket envoyé sur Discord !")

with tab3:
    st.subheader("📊 CLASSEMENTS & FORME")
    l_sel = st.selectbox("LIGUE", list(LEAGUES_DICT.keys()), key="st_l")
    standings = get_api("standings", {"league": LEAGUES_DICT[l_sel], "season": SEASON})
    if standings:
        df = pd.DataFrame([{"Equipe": t['team']['name'], "Pts": t['points'], "Forme": t['form']} for t in standings[0]['league']['standings'][0]])
        st.dataframe(df, use_container_width=True)

st.markdown("""<a href="https://github.com/clementrnx" class="github-link">GITHUB : github.com/clementrnx</a>""", unsafe_allow_html=True)
