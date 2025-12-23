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

# --- CONFIG API & RISK ---
API_KEY = st.secrets["MY_API_KEY"]
BASE_URL = "https://v3.football.api-sports.io/"
HEADERS = {'x-apisports-key': API_KEY}
SEASON = 2025

LEAGUES_DICT = {"La Liga": 140, "Premier League": 39, "Champions League": 2, "Ligue 1": 61, "Serie A": 135, "Bundesliga": 78}

# Les paliers ajustent maintenant le seuil de "Score Élite" (P² * Cote)
RISK_LEVELS = {
    "SAFE": {"elite_min": 0.95, "p_min": 0.85, "kelly": 0.04},
    "MID-SAFE": {"elite_min": 0.85, "p_min": 0.75, "kelly": 0.06},
    "MID": {"elite_min": 0.75, "p_min": 0.65, "kelly": 0.08},
    "MID-AGGRESSIF": {"elite_min": 0.60, "p_min": 0.50, "kelly": 0.12},
    "AGGRESSIF": {"elite_min": 0.45, "p_min": 0.38, "kelly": 0.18}
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
        "p_btts": np.sum(matrix[1:, 1:]), "p_nobtts": 1.0 - np.sum(matrix[1:, 1:]), "matrix": matrix
    }

# --- UI ---
st.title(" CLEMENTRNXX PREDICTOR V5.5")
st.subheader("MODÈLE ÉLITE : MAXIMISATION RENTABILITÉ RÉELLE (P² × Cote)")

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
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric(th, f"{r['p_h']*100:.1f}%")
        m2.metric("MATCH NUL", f"{r['p_n']*100:.1f}%")
        m3.metric(ta, f"{r['p_a']*100:.1f}%")
        m4.metric("BTTS OUI", f"{r['p_btts']*100:.1f}%")
        m5.metric("BTTS NON", f"{r['p_nobtts']*100:.1f}%")

with tab2:
    st.subheader(" GÉNÉRATEUR DE TICKETS À RENTABILITÉ MAXIMALE")
    gc1, gc2, gc3, gc4 = st.columns(4)
    l_scan = gc1.selectbox("LIGUES", ["TOUTES LES LEAGUES"] + list(LEAGUES_DICT.keys()), key="sc_l")
    d_scan = gc2.date_input("DATE", datetime.now(), key="sc_d")
    bank_scan = gc3.number_input("FONDS (€)", value=100.0, key="sc_b")
    scope_scan = gc4.select_slider("DATA", options=["LEAGUE ONLY", "OVER-ALL"], value="OVER-ALL", key="sc_s")
    
    risk_mode = st.select_slider("PHILOSOPHIE DE JEU", options=list(RISK_LEVELS.keys()), value="MID", key="sc_r")
    cfg = RISK_LEVELS[risk_mode]
    
    if st.button("GÉNÉRER LE MEILLEUR TICKET", key="sc_btn"):
        lids = LEAGUES_DICT.values() if l_scan == "TOUTES LES LEAGUES" else [LEAGUES_DICT[l_scan]]
        opps = []
        with st.spinner("Calcul des probabilités d'élite..."):
            for lid in lids:
                fixtures = get_api("fixtures", {"league": lid, "season": SEASON, "date": d_scan.strftime('%Y-%m-%d')})
                for f in fixtures:
                    if f['fixture']['status']['short'] != "NS": continue
                    ah, dh = get_team_stats(f['teams']['home']['id'], lid, scope_scan=="OVER-ALL")
                    aa, da = get_team_stats(f['teams']['away']['id'], lid, scope_scan=="OVER-ALL")
                    lh, la = (ah * da) ** 0.5 * 1.05, (aa * dh) ** 0.5 * 0.95
                    pr = calculate_perfect_probs(lh, la)
                    
                    odds = get_api("odds", {"fixture": f['fixture']['id']})
                    if odds and odds[0]['bookmakers']:
                        for b_type in odds[0]['bookmakers'][0]['bets']:
                            if b_type['name'] == "Match Winner":
                                for o in b_type['values']:
                                    p_val = pr['p_h'] if o['value'] == 'Home' else pr['p_n'] if o['value'] == 'Draw' else pr['p_a']
                                    cote = float(o['odd'])
                                    # LOGIQUE ÉLITE : P² * Cote
                                    elite_score = (p_val ** 2) * cote
                                    if p_val >= cfg['p_min'] and elite_score >= cfg['elite_min']:
                                        opps.append({
                                            "MATCH": f"{f['teams']['home']['name']}-{f['teams']['away']['name']}",
                                            "PARI": o['value'], "COTE": cote, "PROBA": p_val, "SCORE_ELITE": elite_score
                                        })
        
        # Tri par Score Élite (les meilleurs rapports Valeur/Chance en premier)
        final_ticket = sorted(opps, key=lambda x: x['SCORE_ELITE'], reverse=True)
        
        if final_ticket:
            total_odd = np.prod([x['COTE'] for x in final_ticket])
            total_proba = np.prod([x['PROBA'] for x in final_ticket])
            mise = bank_scan * cfg['kelly']
            
            st.markdown(f"""
            <div class='verdict-box'>
                <h2 style='margin:0'>TICKET {risk_mode} GÉNÉRÉ</h2>
                <h1 style='color:#FFD700; font-size:3rem'>@{total_odd:.2f}</h1>
                <p>Probabilité de succès : <b>{total_proba*100:.2f}%</b></p>
                <p>Mise conseillée : <b>{mise:.2f}€</b> | Gain : <b>{mise*total_odd:.2f}€</b></p>
            </div>
            """, unsafe_allow_html=True)
            
            df_disp = pd.DataFrame(final_ticket)
            df_disp['PROBA'] = df_disp['PROBA'].apply(lambda x: f"{x*100:.1f}%")
            st.table(df_disp[['MATCH', 'PARI', 'COTE', 'PROBA', 'SCORE_ELITE']])
        else:
            st.error("Aucun match ne valide les critères d'élite pour ce mode aujourd'hui.")

with tab3:
    st.subheader("📊 CLASSEMENTS")
    l_sel = st.selectbox("LIGUE", list(LEAGUES_DICT.keys()), key="st_l")
    standings = get_api("standings", {"league": LEAGUES_DICT[l_sel], "season": SEASON})
    if standings:
        df = pd.DataFrame([{"Equipe": t['team']['name'], "Pts": t['points'], "Forme": t['form']} for t in standings[0]['league']['standings'][0]])
        st.dataframe(df, use_container_width=True)

st.markdown("""<a href="https://github.com/clementrnx" class="github-link" target="_blank">GITHUB : github.com/clementrnx</a>""", unsafe_allow_html=True)
