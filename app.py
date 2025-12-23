import streamlit as st
import requests
import numpy as np
from scipy.stats import poisson
from datetime import datetime
import pandas as pd
import time

# --- CONFIGURATION CLEMENTRNXX PREDICTOR V5.5 ---
st.set_page_config(page_title="Clementrnxx Predictor V5.5", layout="wide")

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
    div.stButton > button:hover { background: #FFD700 !important; color: black !important; box-shadow: 0 0 30px rgba(255, 215, 0, 0.6); }
    .verdict-box { border: 2px solid #FFD700; padding: 25px; text-align: center; border-radius: 20px; background: rgba(0,0,0,0.8); margin: 20px 0; }
    .score-card { background: rgba(255, 255, 255, 0.07); border: 1px solid rgba(255, 215, 0, 0.4); padding: 15px; border-radius: 12px; text-align: center; }
    .github-link { display: block; text-align: center; color: #FFD700 !important; font-weight: bold; font-size: 1.2rem; text-decoration: none; margin-top: 40px; padding: 20px; border-top: 1px solid rgba(255, 215, 0, 0.2); }
    </style>
""", unsafe_allow_html=True)

# --- CONFIG API & RISK (ASSOUPLI) ---
API_KEY = st.secrets["MY_API_KEY"]
BASE_URL = "https://v3.football.api-sports.io/"
HEADERS = {'x-apisports-key': API_KEY}
SEASON = 2025
LEAGUES_DICT = {"La Liga": 140, "Premier League": 39, "Champions League": 2, "Ligue 1": 61, "Serie A": 135, "Bundesliga": 78}

# Seuils assouplis pour garantir des résultats tout en gardant la logique Elite
RISK_LEVELS = {
    "SAFE": {"elite_min": 0.70, "p_min": 0.75, "kelly": 0.04},
    "MID-SAFE": {"elite_min": 0.60, "p_min": 0.65, "kelly": 0.06},
    "MID": {"elite_min": 0.50, "p_min": 0.55, "kelly": 0.08},
    "MID-AGGRESSIF": {"elite_min": 0.40, "p_min": 0.45, "kelly": 0.12},
    "AGGRESSIF": {"elite_min": 0.25, "p_min": 0.30, "kelly": 0.18}
}

# --- FONCTIONS ---
@st.cache_data(ttl=3600)
def get_api(endpoint, params):
    try:
        r = requests.get(f"{BASE_URL}{endpoint}", headers=HEADERS, params=params, timeout=12)
        return r.json().get('response', [])
    except: return []

def get_team_stats(team_id, league_id, scope_overall):
    params = {"team": team_id, "season": SEASON, "last": 15}
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
        "p_btts": np.sum(matrix[1:, 1:]), "p_nobtts": 1.0 - np.sum(matrix[1:, 1:])
    }

# --- UI ---
st.title(" CLEMENTRNXX PREDICTOR V5.5")

tab1, tab2, tab3 = st.tabs([" ANALYSE 1VS1", " SCANNER DE TICKETS", " STATS"])

with tab1:
    l_name = st.selectbox("LIGUE DU MATCH", list(LEAGUES_DICT.keys()), key="1v1_l")
    scope_1v1 = st.select_slider("MODE DATA SOURCE", options=["LEAGUE ONLY", "OVER-ALL"], value="OVER-ALL", key="1v1_s")
    teams_res = get_api("teams", {"league": LEAGUES_DICT[l_name], "season": SEASON})
    teams = {t['team']['name']: t['team']['id'] for t in teams_res}
    
    if teams:
        c1, c2 = st.columns(2)
        th = c1.selectbox("DOMICILE", sorted(teams.keys()), key="1v1_h")
        ta = c2.selectbox("EXTÉRIEUR", sorted(teams.keys()), key="1v1_a")
        
        if st.button("LANCER L'ANALYSE", key="1v1_btn"):
            att_h, def_h = get_team_stats(teams[th], LEAGUES_DICT[l_name], scope_1v1=="OVER-ALL")
            att_a, def_a = get_team_stats(teams[ta], LEAGUES_DICT[l_name], scope_1v1=="OVER-ALL")
            lh, la = (att_h * def_a) ** 0.5 * 1.05, (att_a * def_h) ** 0.5 * 0.95
            st.session_state.v5_final = {"res": calculate_perfect_probs(lh, la), "th": th, "ta": ta}

    if 'v5_final' in st.session_state:
        r, th, ta = st.session_state.v5_final["res"], st.session_state.v5_final["th"], st.session_state.v5_final["ta"]
        st.markdown("### 📊 Résultats Probabilités")
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric(th, f"{r['p_h']*100:.1f}%")
        m2.metric("MATCH NUL", f"{r['p_n']*100:.1f}%")
        m3.metric(ta, f"{r['p_a']*100:.1f}%")
        m4.metric("BTTS OUI", f"{r['p_btts']*100:.1f}%")
        m5.metric("BTTS NON", f"{r['p_nobtts']*100:.1f}%")

        st.markdown("---")
        st.subheader("💰 CALCULATEUR DE VALUE (Toutes catégories)")
        bankroll = st.number_input("BANQUE (€)", value=100.0, key="1v1_bank")
        
        ic1, ic2, ic3, ic4 = st.columns(4)
        c_h = ic1.number_input(f"Cote {th}", 1.0, key="c_h")
        c_n = ic2.number_input("Cote NUL", 1.0, key="c_n")
        c_a = ic3.number_input(f"Cote {ta}", 1.0, key="c_a")
        c_1n = ic4.number_input(f"{th}/N", 1.0, key="c_1n")

        ic5, ic6, ic7, ic8 = st.columns(4)
        c_n2 = ic5.number_input(f"N/{ta}", 1.0, key="c_n2")
        c_12 = ic6.number_input("12 (Pas Nul)", 1.0, key="c_12")
        c_by = ic7.number_input("BTTS OUI", 1.0, key="c_by")
        c_bn = ic8.number_input("BTTS NON", 1.0, key="c_bn")

        # Analyse automatique des values
        bets = [
            (f"Victoire {th}", c_h, r['p_h']), ("Match Nul", c_n, r['p_n']), (f"Victoire {ta}", c_a, r['p_a']),
            ("Double Chance 1N", c_1n, r['p_1n']), ("Double Chance N2", c_n2, r['p_n2']), ("Double Chance 12", c_12, r['p_12']),
            ("BTTS OUI", c_by, r['p_btts']), ("BTTS NON", c_bn, r['p_nobtts'])
        ]
        
        st.markdown("### 🎯 Verdict Sniping")
        found_value = False
        for name, cote, prob in bets:
            if cote > 1.0:
                ev = cote * prob
                elite = (prob**2) * cote
                if ev > 1.05:
                    st.success(f"**{name}** | EV: {ev:.2f} | Score Élite: {elite:.2f} | Mise suggérée: {(bankroll*0.05):.2f}€")
                    found_value = True
        if not found_value: st.info("Saisissez les cotes de votre bookmaker pour voir les values.")

with tab2:
    st.subheader(" GÉNÉRATEUR DE TICKETS OPTIMISÉS (Scanner Souple)")
    sc1, sc2, sc3 = st.columns([2, 1, 1])
    l_scan = sc1.selectbox("LIGUES", ["TOUTES LES LEAGUES"] + list(LEAGUES_DICT.keys()), key="sc_l")
    d_scan = sc2.date_input("DATE", datetime.now(), key="sc_d")
    risk_mode = sc3.select_slider("MODE", options=list(RISK_LEVELS.keys()), value="MID", key="sc_r")
    
    cfg = RISK_LEVELS[risk_mode]
    
    if st.button("LANCER LE SCANNER ÉLITE", key="sc_btn"):
        lids = LEAGUES_DICT.values() if l_scan == "TOUTES LES LEAGUES" else [LEAGUES_DICT[l_scan]]
        opps = []
        with st.spinner("Analyse des opportunités mondiales..."):
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
                            # Scan assoupli sur plusieurs marchés
                            if market['name'] in ["Match Winner", "Double Chance", "Both Teams Score"]:
                                for o in market['values']:
                                    # Mapping des probabilités
                                    p_val = 0
                                    if o['value'] == 'Home': p_val = pr['p_h']
                                    elif o['value'] == 'Draw': p_val = pr['p_n']
                                    elif o['value'] == 'Away': p_val = pr['p_a']
                                    elif o['value'] == 'Home/Draw': p_val = pr['p_1n']
                                    elif o['value'] == 'Draw/Away': p_val = pr['p_n2']
                                    elif o['value'] == 'Home/Away': p_val = pr['p_12']
                                    elif o['value'] == 'Yes': p_val = pr['p_btts']
                                    elif o['value'] == 'No': p_val = pr['p_nobtts']
                                    
                                    if p_val > 0:
                                        cote = float(o['odd'])
                                        elite = (p_val ** 2) * cote
                                        # Filtre souple mais qualitatif
                                        if p_val >= cfg['p_min'] and elite >= cfg['elite_min']:
                                            opps.append({
                                                "MATCH": f"{f['teams']['home']['name']}-{f['teams']['away']['name']}",
                                                "PARI": f"{market['name']} : {o['value']}",
                                                "COTE": cote,
                                                "PROBA": p_val,
                                                "SCORE": elite
                                            })
        
        final_ticket = sorted(opps, key=lambda x: x['SCORE'], reverse=True)
        
        if final_ticket:
            total_odd = np.prod([x['COTE'] for x in final_ticket[:5]]) # Top 5 pour le ticket
            st.markdown(f"""
            <div class='verdict-box'>
                <h2 style='margin:0'>SÉLECTION ÉLITE {risk_mode}</h2>
                <h1 style='color:#FFD700'>@{total_odd:.2f}</h1>
                <p>Analyse sur {len(final_ticket)} opportunités trouvées.</p>
            </div>
            """, unsafe_allow_html=True)
            st.table(pd.DataFrame(final_ticket).head(10))
        else:
            st.warning("Aucun arbitrage détecté. Essayez un mode plus souple ou une autre date.")

with tab3:
    st.subheader("📊 CLASSEMENTS & FORME")
    l_sel = st.selectbox("LIGUE", list(LEAGUES_DICT.keys()), key="st_l")
    standings = get_api("standings", {"league": LEAGUES_DICT[l_sel], "season": SEASON})
    if standings:
        df = pd.DataFrame([{"Equipe": t['team']['name'], "Pts": t['points'], "Forme": t['form']} for t in standings[0]['league']['standings'][0]])
        st.dataframe(df, use_container_width=True)

st.markdown("""<a href="https://github.com/clementrnx" class="github-link" target="_blank">GITHUB : github.com/clementrnx</a>""", unsafe_allow_html=True)
