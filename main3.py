import streamlit as st
import requests
import numpy as np
from scipy.stats import poisson

st.set_page_config(page_title="iTrOz Predictor Elite", layout="wide")

# --- NOUVELLE CLE API ---
API_KEY = "f088c65ff3ea4ca9b6a29fe1c2429faf"
BASE_URL = "https://v3.football.api-sports.io/"
HEADERS = {'x-apisports-key': API_KEY}

st.markdown("""
    <style>
    .main { background-color: #050505; color: #FFFFFF; }
    .stMetric { background-color: #111111; border: 1px solid #FFD700; padding: 15px; border-radius: 10px; border-left: 5px solid #FFD700; }
    div[data-testid="stMetricValue"] > div { color: #FFD700 !important; font-weight: 800; }
    button { background-color: #FFD700 !important; color: black !important; font-weight: bold; width: 100%; height: 60px; border-radius: 12px; border: none; font-size: 18px; }
    </style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=86400)
def get_teams(league_id):
    try:
        res = requests.get(f"{BASE_URL}teams", headers=HEADERS, params={"league": league_id, "season": 2024}, timeout=5)
        data = res.json().get('response', [])
        if not data: return {}
        return {t['team']['name']: t['team']['id'] for t in data}
    except: return {}

@st.cache_data(ttl=3600)
def get_stats(league_id, team_id):
    try:
        res = requests.get(f"{BASE_URL}teams/statistics", headers=HEADERS, params={"league": league_id, "season": 2024, "team": team_id}, timeout=5)
        return res.json().get('response', {})
    except: return {}

def rho_correction(x, y, lh, la, rho):
    if x == 0 and y == 0: return 1 - (lh * la * rho)
    elif x == 0 and y == 1: return 1 + (lh * rho)
    elif x == 1 and y == 0: return 1 + (la * rho)
    elif x == 1 and y == 1: return 1 - rho
    return 1

st.title("iTrOz Predictor Elite v4")

leagues_cfg = {
    "Premier League": {"id": 39, "weight": 1.25},
    "La Liga": {"id": 140, "weight": 1.15},
    "Serie A": {"id": 135, "weight": 1.10},
    "Bundesliga": {"id": 78, "weight": 1.20},
    "Ligue 1": {"id": 61, "weight": 1.00}
}

col1, col2 = st.columns(2)

with col1:
    l_h_name = st.selectbox("League Home", list(leagues_cfg.keys()), key="l_h")
    team_list_h = get_teams(leagues_cfg[l_h_name]["id"])
    t_h_name = st.selectbox("Team Home", list(team_list_h.keys()) if team_list_h else ["Indisponible"], key="t_h")
    t_h_id = team_list_h.get(t_h_name)

with col2:
    l_a_name = st.selectbox("League Away", list(leagues_cfg.keys()), key="l_a")
    team_list_a = get_teams(leagues_cfg[l_a_name]["id"])
    t_a_name = st.selectbox("Team Away", list(team_list_a.keys()) if team_list_a else ["Indisponible"], key="t_a")
    t_a_id = team_list_a.get(t_a_name)

rho_val = st.sidebar.slider("Ajustement Rho", -0.20, 0.20, -0.06)

if st.button("LANCER L'ALGORITHME PREDICTIF"):
    if t_h_id and t_a_id:
        with st.spinner('Calcul en cours...'):
            s_h = get_stats(leagues_cfg[l_h_name]["id"], t_h_id)
            s_a = get_stats(leagues_cfg[l_a_name]["id"], t_a_id)
            
            if s_h and s_a:
                cs_h = (s_h.get('clean_sheet', {}).get('home', 0) / max(s_h.get('fixtures', {}).get('played', {}).get('home', 1), 1))
                cs_a = (s_a.get('clean_sheet', {}).get('away', 0) / max(s_a.get('fixtures', {}).get('played', {}).get('away', 1), 1))
                
                att_h = float(s_h.get('goals',{}).get('for',{}).get('average',{}).get('home', 1.1))
                def_h = float(s_h.get('goals',{}).get('against',{}).get('average',{}).get('home', 1.0)) * (1 - cs_h * 0.15)
                att_a = float(s_a.get('goals',{}).get('for',{}).get('average',{}).get('away', 1.0))
                def_a = float(s_a.get('goals',{}).get('against',{}).get('average',{}).get('away', 1.1)) * (1 - cs_a * 0.15)

                w_h, w_a = leagues_cfg[l_h_name]["weight"], leagues_cfg[l_a_name]["weight"]
                mu_h = (att_h * w_h) * (def_a / w_a) * 1.05
                mu_a = (att_a * w_a) * (def_h / w_h)

                matrix = np.zeros((9, 9))
                for x in range(9):
                    for y in range(9):
                        matrix[x,y] = poisson.pmf(x, mu_h) * poisson.pmf(y, mu_a) * rho_correction(x, y, mu_h, mu_a, rho_val)
                matrix /= matrix.sum()

                p_h, p_d, p_a = np.sum(np.tril(matrix, -1))*100, np.sum(np.diag(matrix))*100, np.sum(np.triu(matrix, 1))*100

                st.markdown("---")
                c1, c2, c3 = st.columns(3)
                c1.metric(t_h_name, f"{p_h:.1f}%")
                c2.metric("NUL", f"{p_d:.1f}%")
                c3.metric(t_a_name, f"{p_a:.1f}%")

                st.subheader("Top Scores Probables")
                flat_indices = np.argsort(matrix.ravel())[-5:][::-1]
                for idx in flat_indices:
                    h, a = np.unravel_index(idx, matrix.shape)
                    st.write(f"Score {h} - {a} : {matrix[h, a]*100:.1f}%")
            else:
                st.error("Stats indisponibles. Verifie ton quota API.")
    else:
        st.error("Selectionne des equipes valides.")
