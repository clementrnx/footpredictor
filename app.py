import streamlit as st
import requests
import numpy as np
from scipy.stats import poisson
from datetime import datetime, timedelta
import pandas as pd
import time

# --- CONFIGURATION CLEMENTRNXX PREDICTOR V5.5 ---
st.set_page_config(page_title="Clementrnxx Predictor V9.5", layout="wide")

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
    .github-link { display: block; text-align: center; color: #FFD700 !important; font-weight: bold; font-size: 1.2rem; text-decoration: none; margin-top: 40px; padding: 20px; border-top: 1px solid rgba(255, 215, 0, 0.2); }
    </style>
""", unsafe_allow_html=True)

# --- CONFIG API & RISK ---
API_KEY = st.secrets["MY_API_KEY"]
DISCORD_WEBHOOK = "https://discord.com/api/webhooks/1453026279275106355/gbYAwBRntm1FCoqoBTz5lj1SCe2ijyeHHYoe4CFYwpzOw2DO-ozcCsgkK_53HhB-kFGE"
BASE_URL = "https://v3.football.api-sports.io/"
HEADERS = {'x-apisports-key': API_KEY}
SEASON = 2025

LEAGUES_DICT = {"La Liga": 140, "Premier League": 39, "Champions League": 2, "Ligue 1": 61, "Serie A": 135, "Bundesliga": 78}
RISK_LEVELS = {
    "SAFE": {"p": 0.82, "ev": 1.02, "kelly": 0.02},
    "MID-SAFE": {"p": 0.74, "ev": 1.05, "kelly": 0.05},
    "MID": {"p": 0.64, "ev": 1.08, "kelly": 0.08},
    "MID-AGGRESSIF": {"p": 0.52, "ev": 1.12, "kelly": 0.12},
    "AGGRESSIF": {"p": 0.42, "ev": 1.15, "kelly": 0.20}
}

# --- FONCTIONS CORE ---
@st.cache_data(ttl=3600)
def get_api(endpoint, params):
    try:
        r = requests.get(f"{BASE_URL}{endpoint}", headers=HEADERS, params=params, timeout=12)
        if r.status_code == 429: time.sleep(1); return get_api(endpoint, params)
        return r.json().get('response', [])
    except: return []

def get_team_stats(team_id, league_id, scope_overall):
    params = {"team": team_id, "season": SEASON, "last": 15}
    if not scope_overall: params["league"] = league_id
    f = get_api("fixtures", params)
    if not f: return 1.2, 1.2
    scored, conceded = [], []
    for m in f:
        if m['goals']['home'] is not None:
            is_home = m['teams']['home']['id'] == team_id
            scored.append(m['goals']['home'] if is_home else m['goals']['away'])
            conceded.append(m['goals']['away'] if is_home else m['goals']['home'])
    if not scored: return 1.2, 1.2
    weights = [0.95 ** i for i in range(len(scored))]
    sum_w = sum(weights)
    return sum(s * w for s, w in zip(scored, weights)) / sum_w, sum(c * w for c, w in zip(conceded, weights)) / sum_w

def calculate_perfect_probs(lh, la):
    rho = -0.11
    matrix = np.zeros((10, 10))
    for x in range(10):
        for y in range(10):
            prob = poisson.pmf(x, lh) * poisson.pmf(y, la)
            adj = 1.0
            if x == 0 and y == 0: adj = 1 - (lh * la * rho)
            elif x == 0 and y == 1: adj = 1 + (lh * rho)
            elif x == 1 and y == 0: adj = 1 + (la * rho)
            elif x == 1 and y == 1: adj = 1 - rho
            matrix[x, y] = max(0, prob * adj)
    if matrix.sum() > 0: matrix /= matrix.sum()
    p_h, p_n, p_a = np.sum(np.tril(matrix, -1)), np.sum(np.diag(matrix)), np.sum(np.triu(matrix, 1))
    return {"p_h": p_h, "p_n": p_n, "p_a": p_a, "p_1n": p_h+p_n, "p_n2": p_n+p_a, "p_12": p_h+p_a, "p_btts": np.sum(matrix[1:, 1:]), "p_nobtts": 1.0 - np.sum(matrix[1:, 1:]), "matrix": matrix}

def send_to_discord(ticket, total_odd, mode):
    matches_text = "\n".join([f"🔹 {t['MATCH']} : **{t['PARI']}** (@{t['COTE']}) - {t['PROBA']*100:.1f}%" for t in ticket])
    payload = {"embeds": [{"title": f"🚀 TICKET MAX GAIN - MODE {mode}", "description": f"{matches_text}\n\n🔥 **COTE TOTALE : @{total_odd:.2f}**", "color": 16766720}]}
    requests.post(DISCORD_WEBHOOK, json=payload)

# --- NAVIGATION ---
st.title(" CLEMENTRNXX PREDICTOR V5.5")
tab1, tab2, tab3 = st.tabs([" ANALYSE 1VS1", " SCANNER DE TICKETS", " STATS"])

with tab1:
    l_name = st.selectbox("LIGUE DU MATCH", list(LEAGUES_DICT.keys()), key="l_1v1")
    scope_1v1 = st.select_slider("MODE DATA SOURCE", options=["LEAGUE ONLY", "OVER-ALL"], value="OVER-ALL", key="s_1v1")
    teams_res = get_api("teams", {"league": LEAGUES_DICT[l_name], "season": SEASON})
    teams = {t['team']['name']: t['team']['id'] for t in teams_res}
    
    if teams:
        c1, c2 = st.columns(2)
        team_h = c1.selectbox("DOMICILE", sorted(teams.keys()), key="th_1v1")
        team_a = c2.selectbox("EXTÉRIEUR", sorted(teams.keys()), key="ta_1v1")
        if st.button("LANCER L'ANALYSE", key="btn_1v1"):
            att_h, def_h = get_team_stats(teams[team_h], LEAGUES_DICT[l_name], scope_1v1=="OVER-ALL")
            att_a, def_a = get_team_stats(teams[team_a], LEAGUES_DICT[l_name], scope_1v1=="OVER-ALL")
            lh, la = (att_h * def_a) ** 0.5 * 1.08, (att_a * def_h) ** 0.5 * 0.92
            st.session_state.v5_final = {"res": calculate_perfect_probs(lh, la), "th": team_h, "ta": team_a}

    if 'v5_final' in st.session_state:
        r, th, ta = st.session_state.v5_final["res"], st.session_state.v5_final["th"], st.session_state.v5_final["ta"]
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric(th, f"{r['p_h']*100:.1f}%")
        m2.metric("MATCH NUL", f"{r['p_n']*100:.1f}%")
        m3.metric(ta, f"{r['p_a']*100:.1f}%")
        m4.metric("BTTS OUI", f"{r['p_btts']*100:.1f}%")
        m5.metric("BTTS NON", f"{r['p_nobtts']*100:.1f}%")

        st.subheader(" MODULE BET")
        bc1, bc2 = st.columns([2, 1])
        with bc2:
            bankroll = st.number_input("FOND DISPONIBLE (€)", value=100.0, step=10.0, key="bk_1v1")
            risk_1v1 = st.selectbox("MODE DE RISQUE", list(RISK_LEVELS.keys()), index=2, key="rm_1v1")
        
        with bc1:
            i1, i2, i3, i4 = st.columns(4)
            c_h = i1.number_input(f"Cote {th}", value=1.0, key="v_h")
            c_hn = i2.number_input(f"{th}/N", value=1.0, key="v_hn")
            c_n2 = i3.number_input(f"N/{ta}", value=1.0, key="v_n2")
            c_a = i4.number_input(f"Cote {ta}", value=1.0, key="v_a")
            i5, i6, i7 = st.columns(3)
            c_12 = i5.number_input(f"{th}/{ta}", value=1.0, key="v_12")
            c_by = i6.number_input("BTTS OUI", value=1.0, key="v_by")
            c_bn = i7.number_input("BTTS NON", value=1.0, key="v_bn")

        cfg = RISK_LEVELS[risk_1v1]
        bets = [(f"Victoire {th}", c_h, r['p_h']), (f"Double Chance {th}/N", c_hn, r['p_1n']), (f"Double Chance N/{ta}", c_n2, r['p_n2']), (f"Victoire {ta}", c_a, r['p_a']), (f"Issue {th}/{ta}", c_12, r['p_12']), ("BTTS OUI", c_by, r['p_btts']), ("BTTS NON", c_bn, r['p_nobtts'])]
        
        st.markdown("### 📋 VERDICT ALGORITHME")
        found = False
        for name, cote, prob in bets:
            if cote > 1.0 and prob >= cfg['p'] and (cote * prob) >= cfg['ev']:
                st.success(f"✅ **CONSEILLÉ : {name}** | Cote: @{cote} | Confiance: {prob*100:.1f}% | Mise: {bankroll*cfg['kelly']:.2f}€")
                found = True
        if not found: st.warning("⚠️ AUCUN PARI NE CORRESPOND À VOS CRITÈRES.")

with tab2:
    st.subheader(" GÉNÉRATEUR DE TICKETS - MAX GAIN OPTIMISÉ")
    gc1, gc2, gc3, gc4 = st.columns(4)
    l_scan = gc1.selectbox("CHAMPIONNAT", ["TOUTES LES LEAGUES"] + list(LEAGUES_DICT.keys()), key="l_scan")
    d_range = gc2.date_input("PÉRIODE", [datetime.now(), datetime.now()], key="d_scan_range")
    bank_scan = gc3.number_input("FOND DISPONIBLE (€) ", value=100.0, key="b_scan_input")
    max_legs = gc4.slider("NB MATCHS MAX", 1, 30, 3, key="m_legs_scan")
    
    scope_scan = st.select_slider("DATA SCAN", options=["LEAGUE ONLY", "OVER-ALL"], value="OVER-ALL", key="sc_scan")
    selected_markets = st.multiselect("MARCHÉS", ["ISSUE SIMPLE", "DOUBLE CHANCE", "BTTS (OUI/NON)"], default=["ISSUE SIMPLE", "DOUBLE CHANCE"], key="mkt_scan")
    risk_mode = st.select_slider("RISQUE (SEUIL SURVIE)", options=["SAFE", "MID-SAFE", "MID", "MID-AGGRESSIF", "AGGRESSIF"], value="MID", key="rk_scan")
    risk_cfg = RISK_LEVELS[risk_mode]
    
    if st.button("GÉNÉRER LE TICKET PARFAIT (MAX GAIN)", key="btn_gen"):
        if not isinstance(d_range, (list, tuple)) or len(d_range) < 2: st.stop()
        
        date_list = pd.date_range(start=d_range[0], end=d_range[1]).tolist()
        lids = LEAGUES_DICT.values() if l_scan == "TOUTES LES LEAGUES" else [LEAGUES_DICT[l_scan]]
        all_opps = []
        progress_bar = st.progress(0)
        
        for idx_d, current_date in enumerate(date_list):
            date_str = current_date.strftime('%Y-%m-%d')
            for lid in lids:
                fixtures = get_api("fixtures", {"league": lid, "season": SEASON, "date": date_str})
                for f in fixtures:
                    if f['fixture']['status']['short'] != "NS": continue
                    att_h, def_h = get_team_stats(f['teams']['home']['id'], lid, scope_scan=="OVER-ALL")
                    att_a, def_a = get_team_stats(f['teams']['away']['id'], lid, scope_scan=="OVER-ALL")
                    lh, la = (att_h * def_a) ** 0.5 * 1.08, (att_a * def_h) ** 0.5 * 0.92
                    pr = calculate_perfect_probs(lh, la)
                    h_n, a_n = f['teams']['home']['name'], f['teams']['away']['name']

                    tests = []
                    if "ISSUE SIMPLE" in selected_markets:
                        tests += [(h_n, pr['p_h'], "Match Winner", "Home"), (a_n, pr['p_a'], "Match Winner", "Away")]
                    if "DOUBLE CHANCE" in selected_markets:
                        tests += [(f"{h_n}/N", pr['p_1n'], "Double Chance", "Home/Draw"), (f"N/{a_n}", pr['p_n2'], "Double Chance", "Draw/Away")]
                    if "BTTS (OUI/NON)" in selected_markets:
                        tests += [("BTTS OUI", pr['p_btts'], "Both Teams Score", "Yes"), ("BTTS NON", pr['p_nobtts'], "Both Teams Score", "No")]

                    if tests:
                        odds_res = get_api("odds", {"fixture": f['fixture']['id']})
                        if odds_res:
                            for lbl, p, m_n, m_v in tests:
                                if p >= 0.50:
                                    for bet in odds_res[0]['bookmakers'][0]['bets']:
                                        if bet['name'] == m_n:
                                            for val in bet['values']:
                                                if val['value'] == m_v:
                                                    cote = float(val['odd'])
                                                    all_opps.append({"MATCH": f"{h_n} - {a_n}", "PARI": lbl, "COTE": cote, "PROBA": p, "EV": cote * p})
            progress_bar.progress((idx_d + 1) / len(date_list))

        # --- LOGIQUE TICKET MULTI-MATCHS MAX GAIN ---
        all_opps = sorted(all_opps, key=lambda x: x['EV'], reverse=True)
        final_selection, seen_matches, pdv = [], set(), 1.0
        seuil_survie = risk_cfg['p']
        
        for o in all_opps:
            if len(final_selection) >= max_legs: break
            if o['MATCH'] not in seen_matches:
                if (pdv * o['PROBA']) >= seuil_survie:
                    final_selection.append(o)
                    seen_matches.add(o['MATCH'])
                    pdv *= o['PROBA']

        if final_selection:
            total_odd = np.prod([x['COTE'] for x in final_selection])
            st.markdown(f"""
                <div class='verdict-box'>
                    <h2>🔥 TICKET MAX GAIN</h2>
                    <p style='font-size:1.2rem;'>Survie Ticket : <b>{pdv*100:.1f}%</b> (Seuil: {seuil_survie*100:.0f}%)</p>
                    <p>Cote Totale : <b>@{total_odd:.2f}</b> | Mise : <b>{(bank_scan * risk_cfg['kelly']):.2f}€</b></p>
                </div>
            """, unsafe_allow_html=True)
            st.table(pd.DataFrame(final_selection)[["MATCH", "PARI", "COTE", "PROBA"]])
            send_to_discord(final_selection, total_odd, risk_mode)
        else: st.error("Impossible de construire un ticket respectant ce seuil de survie.")

with tab3:
    st.subheader("📊 CLASSEMENTS")
    l_sel = st.selectbox("LIGUE", list(LEAGUES_DICT.keys()), key="sel_tab3")
    standings = get_api("standings", {"league": LEAGUES_DICT[l_sel], "season": SEASON})
    if standings:
        df = pd.DataFrame([{"Equipe": t['team']['name'], "Pts": t['points'], "Forme": t['form']} for t in standings[0]['league']['standings'][0]])
        st.dataframe(df, use_container_width=True)

st.markdown("""<a href="https://github.com/clementrnx" class="github-link" target="_blank">GITHUB : github.com/clementrnx</a>""", unsafe_allow_html=True)
