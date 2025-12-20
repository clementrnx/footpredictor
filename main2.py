import streamlit as st
import requests
import numpy as np
from scipy.stats import poisson

st.set_page_config(page_title="iTrOz Predictor | Deep Stats", layout="wide")

API_KEY = "f30f164b1e1b47cfb7ede5d459f5ab54"
BASE_URL = "https://v3.football.api-sports.io/"
HEADERS = {'x-apisports-key': API_KEY}

st.markdown("""
    <style>
    .main { background-color: #050505; color: #FFFFFF; }
    .stMetric { background-color: #111111; border: 1px solid #FFD700; padding: 10px; border-radius: 10px; }
    div[data-testid="stMetricValue"] > div { color: #FFD700 !important; }
    button { background-color: #FFD700 !important; color: black !important; font-weight: bold; width: 100%; height: 50px; border-radius: 10px; border: none; }
    .stProgress > div > div > div > div { background-color: #FFD700; }
    </style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=3600)
def fetch_api(endpoint, params):
    res = requests.get(f"{BASE_URL}{endpoint}", headers=HEADERS, params=params)
    return res.json()['response']

def rho_correction(x, y, lh, la, rho):
    if x == 0 and y == 0: return 1 - (lh * la * rho)
    elif x == 0 and y == 1: return 1 + (lh * rho)
    elif x == 1 and y == 0: return 1 + (la * rho)
    elif x == 1 and y == 1: return 1 - rho
    return 1

st.title("iTrOz Predictor Deep-Analytics")

leagues = {"PL": 39, "La Liga": 140, "Serie A": 135, "Bundesliga": 78, "Ligue 1": 61}

col1, col2 = st.columns(2)
with col1:
    l_h = st.selectbox("League Home", list(leagues.keys()))
    teams_h = fetch_api("teams", {"league": leagues[l_h], "season": 2024})
    team_list_h = {t['team']['name']: t['team']['id'] for t in teams_h}
    t_h_name = st.selectbox("Team Home", list(team_list_h.keys()))
    t_h_id = team_list_h[t_h_name]

with col2:
    l_a = st.selectbox("League Away", list(leagues.keys()))
    teams_a = fetch_api("teams", {"league": leagues[l_a], "season": 2024})
    team_list_a = {t['team']['name']: t['team']['id'] for t in teams_a}
    t_a_name = st.selectbox("Team Away", list(team_list_a.keys()))
    t_a_id = team_list_a[t_a_name]

if st.button("CALCULER LA REALITE STATISTIQUE"):
    s_h_data = fetch_api("teams/statistics", {"league": leagues[l_h], "season": 2024, "team": t_h_id})
    s_a_data = fetch_api("teams/statistics", {"league": leagues[l_a], "season": 2024, "team": t_a_id})

    form_h = len([x for x in s_h_data['form'][-5:] if x == 'W']) / 5
    form_a = len([x for x in s_a_data['form'][-5:] if x == 'W']) / 5

    att_h = float(s_h_data['goals']['for']['average']['home']) * (1 + form_h * 0.1)
    def_h = float(s_h_data['goals']['against']['average']['home'])
    att_a = float(s_a_data['goals']['for']['average']['away']) * (1 + form_a * 0.1)
    def_a = float(s_a_data['goals']['against']['average']['away'])

    mu_h = att_h * (def_a / 1.5)
    mu_a = att_a * (def_h / 1.5)

    matrix = np.zeros((10, 10))
    rho = -0.08
    for x in range(10):
        for y in range(10):
            matrix[x,y] = poisson.pmf(x, mu_h) * poisson.pmf(y, mu_a) * rho_correction(x, y, mu_h, mu_a, rho)
    matrix /= matrix.sum()

    p_h, p_d, p_a = np.sum(np.tril(matrix, -1))*100, np.sum(np.diag(matrix))*100, np.sum(np.triu(matrix, 1))*100

    st.markdown("---")
    res1, res2, res3 = st.columns(3)
    res1.metric(t_h_name, f"{p_h:.1f}%")
    res2.metric("Nul", f"{p_d:.1f}%")
    res3.metric(t_a_name, f"{p_a:.1f}%")

    st.subheader("Top 3 Scores Probables")
    flat_indices = np.argsort(matrix.ravel())[-3:][::-1]
    for idx in flat_indices:
        h, a = np.unravel_index(idx, matrix.shape)
        st.write(f"Score {h} - {a} : {matrix[h, a]*100:.1f}%")
