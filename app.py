
import streamlit as st
import requests
import numpy as np
from scipy.stats import poisson
from datetime import datetime, timedelta
import pandas as pd

# --- CONFIGURATION CLEMENTRNXX PREDICTOR V12.0 ---
st.set_page_config(page_title="Clementrnxx Predictor V12.0", layout="wide")

DISCORD_WEBHOOK_URL = "https://discord.com/api/webhooks/1453026279275106355/gbYAwBRntm1FCoqoBTz5lj1SCe2ijyeHHYoe4CFYwpzOw2DO-ozcCsgkK_53HhB-kFGE"

st.markdown("""
    <style>
    .stApp { background-image: url("https://media.giphy.com/media/VZrfUvQjXaGEQy1RSn/giphy.gif"); background-size: cover; background-attachment: fixed; }
    .stApp > div:first-child { background-color: rgba(0, 0, 0, 0.95); }
    h1, h2, h3, p, span, label { color: #FFD700 !important; font-family: 'Monospace', sans-serif; }
    .section-box { border: 2px solid #FFD700; padding: 20px; border-radius: 15px; background: rgba(0,0,0,0.8); margin-bottom: 20px; }
    .audit-gold { border: 3px solid #FFD700; background: linear-gradient(145deg, #1a1a1a, #000); padding: 25px; border-radius: 15px; text-align: center; }
    .stake-val { font-size: 2rem; color: #00FF00 !important; font-weight: bold; }
    div.stButton > button { background: #FFD700 !important; color: black !important; font-weight: 900; border-radius: 10px; height: 3em; width: 100%; }
    </style>
""", unsafe_allow_html=True)

# --- CONFIG & MATHS ---
API_KEY = st.secrets["MY_API_KEY"]
BASE_URL = "https://v3.football.api-sports.io/"
HEADERS = {'x-apisports-key': API_KEY}
SEASON = 2025
LEAGUES_DICT = {"La Liga": 140, "Premier League": 39, "Champions League": 2, "Ligue 1": 61, "Serie A": 135, "Bundesliga": 78}

RISK_LEVELS = {
    "ULTRA-SAFE": {"elite": 0.90, "p_min": 0.88, "kelly": 0.05},
    "SAFE": {"elite": 0.75, "p_min": 0.80, "kelly": 0.10},
    "MID-SAFE": {"elite": 0.65, "p_min": 0.70, "kelly": 0.15},
    "MID": {"elite": 0.55, "p_min": 0.60, "kelly": 0.20},
    "MID-AGGRESSIF": {"elite": 0.45, "p_min": 0.50, "kelly": 0.25},
    "JACKPOT": {"elite": 0.30, "p_min": 0.35, "kelly": 0.35}
}

def calculate_probs(lh, la):
    matrix = np.zeros((10, 10))
    for x in range(10):
        for y in range(10): matrix[x, y] = poisson.pmf(x, lh) * poisson.pmf(y, la)
    matrix /= matrix.sum()
    return {
        "Home": np.sum(np.tril(matrix, -1)), "Draw": np.sum(np.diag(matrix)), "Away": np.sum(np.triu(matrix, 1)),
        "1N": np.sum(np.tril(matrix, -1)) + np.sum(np.diag(matrix)),
        "N2": np.sum(np.diag(matrix)) + np.sum(np.triu(matrix, 1)),
        "12": np.sum(np.tril(matrix, -1)) + np.sum(np.triu(matrix, 1)),
        "BTTS_Yes": np.sum(matrix[1:, 1:]), "BTTS_No": 1.0 - np.sum(matrix[1:, 1:])
    }

@st.cache_data(ttl=3600)
def get_api(endpoint, params):
    try: return requests.get(f"{BASE_URL}{endpoint}", headers=HEADERS, params=params, timeout=12).json().get('response', [])
    except: return []

def get_team_stats(team_id, league_id, scope_overall, last_n=15):
    params = {"team": team_id, "season": SEASON, "last": last_n}
    if not scope_overall: params["league"] = league_id
    f = get_api("fixtures", params)
    if not f: return 1.3, 1.3
    scored = [m['goals']['home'] if m['teams']['home']['id'] == team_id else m['goals']['away'] for m in f if m['goals']['home'] is not None]
    conceded = [m['goals']['away'] if m['teams']['home']['id'] == team_id else m['goals']['home'] for m in f if m['goals']['home'] is not None]
    weights = [0.96 ** i for i in range(len(scored))]
    return sum(s * w for s, w in zip(scored, weights)) / sum(weights), sum(c * w for c, w in zip(conceded, weights)) / sum(weights)

# --- UI SIDEBAR ---
with st.sidebar:
    st.header("💰 GESTION BANKROLL")
    bankroll = st.number_input("Capital (€)", 10.0, 100000.0, 1000.0)
    risk_choice = st.select_slider("MODE DE RISQUE", options=list(RISK_LEVELS.keys()), value="MID")
    st.write(f"Fraction de Kelly : {RISK_LEVELS[risk_choice]['kelly']*100}%")

tab1, tab2 = st.tabs(["🎯 ANALYSEUR 1VS1 PRO", "📡 GÉNÉRATEUR MULTI-DATES"])

with tab1:
    st.subheader("🛠 CONFIGURATION DU MATCH")
    c1, c2, c3 = st.columns([2, 2, 1])
    l_name = c1.selectbox("LIGUE", list(LEAGUES_DICT.keys()))
    scope = c2.select_slider("SCOPE DATA", ["LEAGUE", "OVER-ALL"], "OVER-ALL")
    n_val = c3.number_input("LAST N", 5, 50, 15)
    
    teams = {t['team']['name']: t['team']['id'] for t in get_api("teams", {"league": LEAGUES_DICT[l_name], "season": SEASON})}
    if teams:
        col_t1, col_t2 = st.columns(2)
        th, ta = col_t1.selectbox("DOMICILE", sorted(teams.keys())), col_t2.selectbox("EXTÉRIEUR", sorted(teams.keys()))
        
        if st.button("LANCER L'AUDIT COMPLET"):
            ah, dh = get_team_stats(teams[th], LEAGUES_DICT[l_name], scope=="OVER-ALL", n_val)
            aa, da = get_team_stats(teams[ta], LEAGUES_DICT[l_name], scope=="OVER-ALL", n_val)
            lh, la = (ah * da) ** 0.5 * 1.05, (aa * dh) ** 0.5 * 0.95
            st.session_state.v12 = {"res": calculate_probs(lh, la), "th": th, "ta": ta}

    if 'v12' in st.session_state:
        r, th_n, ta_n = st.session_state.v12["res"], st.session_state.v12["th"], st.session_state.v12["ta"]
        
        # --- 1. CATEGORIE ANALYSE ---
        st.markdown("<div class='section-box'><h3>📊 1. ANALYSE DES PROBABILITÉS</h3>", unsafe_allow_html=True)
        m = st.columns(4)
        m[0].metric(th_n, f"{r['Home']:.1%}"); m[1].metric("NUL", f"{r['Draw']:.1%}"); m[2].metric(ta_n, f"{r['Away']:.1%}"); m[3].metric("BTTS OUI", f"{r['BTTS_Yes']:.1%}")
        st.markdown("</div>", unsafe_allow_html=True)

        # --- 2. CATEGORIE BET ---
        st.markdown("<div class='section-box'><h3>💰 2. ZONE DE BET (COTES)</h3>", unsafe_allow_html=True)
        b1, b2, b3 = st.columns(3); ch = b1.number_input(f"Cote {th_n}", 1.0, key="v12_ch"); cn = b2.number_input("Cote N", 1.0, key="v12_cn"); ca = b3.number_input(f"Cote {ta_n}", 1.0, key="v12_ca")
        b4, b5, b6 = st.columns(3); c1n = b4.number_input("Cote 1N", 1.0, key="v12_c1n"); cn2 = b5.number_input("Cote N2", 1.0, key="v12_cn2"); cby = b6.number_input("Cote BTTS OUI", 1.0, key="v12_cby")
        st.markdown("</div>", unsafe_allow_html=True)

        # --- 3. CATEGORIE AUDIT ---
        st.markdown("<div class='audit-gold'><h3>🛡️ 3. AUDIT DE MISE ÉLITE</h3>", unsafe_allow_html=True)
        bets = [
            (f"Victoire {th_n}", ch, r['Home']), ("Match Nul", cn, r['Draw']), (f"Victoire {ta_n}", ca, r['Away']),
            ("Double Chance 1N", c1n, r['1N']), ("Double Chance N2", cn2, r['N2']), ("BTTS OUI", cby, r['BTTS_Yes'])
        ]
        
        has_value = False
        for name, cote, prob in bets:
            if cote > 1.05:
                ev = cote * prob
                if ev > 1.08:
                    has_value = True
                    b_val = cote - 1
                    f_kelly = (prob * b_val - (1 - prob)) / b_val
                    mise = round(max(0, f_kelly * bankroll * RISK_LEVELS[risk_choice]['kelly']), 2)
                    st.markdown(f"🔥 **VALUE DÉTECTÉE : {name}**")
                    st.markdown(f"EV: {ev:.2f} | Confiance: {(prob**2*cote):.2f}")
                    st.markdown(f"<p class='stake-val'>MISE CONSEILLÉE : {mise} €</p>", unsafe_allow_html=True)
        
        if not has_value: st.warning("⚠️ Aucun pari rentable détecté pour ce match.")
        st.markdown("</div>", unsafe_allow_html=True)

with tab2:
    st.subheader("📡 SCANNER DE COMBINÉS MULTI-MARCHÉS")
    sd, ed, sl = st.columns([2, 2, 2])
    date_start, date_end = sd.date_input("DÉBUT", datetime.now()), ed.date_input("FIN", datetime.now() + timedelta(days=3))
    ligue_scan = sl.selectbox("LIGUES", ["TOUTES"] + list(LEAGUES_DICT.keys()))
    
    m1, m2 = st.columns(2)
    mkt_active = m1.multiselect("MARCHÉS", ["1N2", "Double Chance", "BTTS"], default=["1N2", "BTTS"])
    max_matches = m2.number_input("MATCHS MAX", 1, 10, 5)

    if st.button("🔥 GÉNÉRER LE TICKET SUR LA PÉRIODE"):
        results = []
        cfg = RISK_LEVELS[risk_choice]
        target_lids = LEAGUES_DICT.values() if ligue_scan == "TOUTES" else [LEAGUES_DICT[ligue_scan]]
        date_range = [date_start + timedelta(days=i) for i in range((date_end - date_start).days + 1)]
        
        for d in date_range:
            for lid in target_lids:
                fixtures = get_api("fixtures", {"league": lid, "season": SEASON, "date": d.strftime('%Y-%m-%d')})
                for f in fixtures:
                    if f['fixture']['status']['short'] != "NS": continue
                    ah, dh = get_team_stats(f['teams']['home']['id'], lid, True, 15)
                    aa, da = get_team_stats(f['teams']['away']['id'], lid, True, 15)
                    pr = calculate_probs((ah * da)**0.5, (aa * dh)**0.5)
                    
                    odds = get_api("odds", {"fixture": f['fixture']['id']})
                    if odds and odds[0]['bookmakers']:
                        for b in odds[0]['bookmakers'][0]['bets']:
                            if (b['name'] == "Match Winner" and "1N2" in mkt_active) or \
                               (b['name'] == "Double Chance" and "Double Chance" in mkt_active) or \
                               (b['name'] == "Both Teams Score" and "BTTS" in mkt_active):
                                for o in b['values']:
                                    label = o['value']
                                    # Mapping
                                    p = 0
                                    if label == "Home": p = pr['Home']
                                    elif label == "Draw": p = pr['Draw']
                                    elif label == "Away": p = pr['Away']
                                    elif label == "Home/Draw": p = pr['1N']
                                    elif label == "Draw/Away": p = pr['N2']
                                    elif label == "Yes": p = pr['BTTS_Yes']
                                    
                                    if p >= cfg['p_min']:
                                        score = (p**2) * float(o['odd'])
                                        if score >= cfg['elite']:
                                            results.append({"Date": d.strftime('%d/%m'), "Match": f"{f['teams']['home']['name']} - {f['teams']['away']['name']}", "Pari": f"{b['name']}: {label}", "Cote": float(o['odd']), "Score": score})

        final_ticket = sorted(results, key=lambda x: x['Score'], reverse=True)[:max_matches]
        if final_ticket:
            st.table(pd.DataFrame(final_ticket))
            tc = np.prod([x['Cote'] for x in final_ticket])
            st.metric("COTE TOTALE", f"@{tc:.2f}")
            # Envoi Discord
            requests.post(DISCORD_WEBHOOK_URL, json={"embeds": [{"title": "V12 ELITE TICKET", "description": f"Ticket généré pour la période {date_start} au {date_end}\nCote: @{tc:.2f}", "color": 16766720}]})
