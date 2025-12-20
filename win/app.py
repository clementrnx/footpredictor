import streamlit as st
import requests
import numpy as np
from scipy.stats import poisson, entropy
import plotly.express as px

st.set_page_config(page_title="iTrOz Predictor ELITE PRO", layout="wide")

API_KEY = ""
BASE_URL = "https://v3.football.api-sports.io/"
HEADERS = {'x-apisports-key': API_KEY}
SEASON_ACTUELLE = 2025
SEASON_PRECEDENTE = 2024

st.markdown("""
    <style>
    .main { background-color: #0A0A0A; color: #E0E0E0; font-family: 'Inter', sans-serif; }
    .stMetric { background-color: #161616; border: 1px solid #FFD700; padding: 20px; border-radius: 8px; }
    div[data-testid="stMetricValue"] > div { color: #FFD700 !important; font-size: 32px; font-weight: 700; }
    .stButton>button { background: #FFD700; color: #000; font-weight: 700; width: 100%; border-radius: 6px; height: 50px; }
    .factor-card { background: #161616; border-left: 4px solid #FFD700; padding: 15px; margin: 10px 0; border-radius: 4px; }
    .score-highlight { background: #1A1A1A; padding: 10px; border-radius: 4px; border: 1px solid #333; margin-bottom: 5px; }
    </style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=3600)
def get_api(endpoint, params):
    try:
        r = requests.get(f"{BASE_URL}{endpoint}", headers=HEADERS, params=params, timeout=10)
        res = r.json().get('response', [])
        if endpoint == "teams/statistics":
            return res if isinstance(res, dict) else (res[0] if res else {})
        return res
    except: return []

def extraire_puissance(stats):
    if not stats or 'goals' not in stats: return 1.35, 1.35
    att = float(stats['goals']['for']['average']['total'] or 1.35)
    defe = float(stats['goals']['against']['average']['total'] or 1.35)
    return att, defe

st.title("iTrOz Predictor ELITE PRO")
st.caption("Analyse prédictive de haute précision")

LEAGUES = {"Premier League": 39, "La Liga": 140, "Serie A": 135, "Bundesliga": 78, "Ligue 1": 61}
nom_ligue = st.selectbox("Championnat", list(LEAGUES.keys()))
id_ligue = LEAGUES[nom_ligue]

data_equipes = get_api("teams", {"league": id_ligue, "season": SEASON_ACTUELLE})
equipes = {t['team']['name']: t['team']['id'] for t in data_equipes} if data_equipes else {}

if equipes:
    col_a, col_b = st.columns(2)
    dom_nom = col_a.selectbox("Équipe Domicile", sorted(equipes.keys()))
    ext_nom = col_b.selectbox("Équipe Extérieur", sorted(equipes.keys()), index=1)

    if st.button("EXECUTER L'ANALYSE"):
        with st.spinner("Calcul des probabilités réelles..."):
            id_dom, id_ext = equipes[dom_nom], equipes[ext_nom]
            
            s25_dom = get_api("teams/statistics", {"league": id_ligue, "season": SEASON_ACTUELLE, "team": id_dom})
            s25_ext = get_api("teams/statistics", {"league": id_ligue, "season": SEASON_ACTUELLE, "team": id_ext})
            s24_dom = get_api("teams/statistics", {"league": id_ligue, "season": SEASON_PRECEDENTE, "team": id_dom})
            s24_ext = get_api("teams/statistics", {"league": id_ligue, "season": SEASON_PRECEDENTE, "team": id_ext})

            att25_dom, def25_dom = extraire_puissance(s25_dom)
            att25_ext, def25_ext = extraire_puissance(s25_ext)
            att24_dom, def24_dom = extraire_puissance(s24_dom)
            att24_ext, def24_ext = extraire_puissance(s24_ext)

            att_final_dom = (att25_dom * 0.7) + (att24_dom * 0.3)
            def_final_dom = (def25_dom * 0.7) + (def24_dom * 0.3)
            att_final_ext = (att25_ext * 0.7) + (att24_ext * 0.3)
            def_final_ext = (def25_ext * 0.7) + (def24_ext * 0.3)

            lh = (att_final_dom * def_final_ext / 1.35) * 1.10
            la = (att_final_ext * def_final_dom / 1.35)

            matrix = np.zeros((8, 8))
            for x in range(8):
                for y in range(8):
                    matrix[x, y] = poisson.pmf(x, lh) * poisson.pmf(y, la)
            matrix /= matrix.sum()

            p_dom = np.sum(np.tril(matrix, -1)) * 100
            p_nul = np.sum(np.diag(matrix)) * 100
            p_ext = np.sum(np.triu(matrix, 1)) * 100

            st.divider()
            r1, r2, r3 = st.columns(3)
            r1.metric(dom_nom, f"{p_dom:.1f}%")
            r2.metric("MATCH NUL", f"{p_nul:.1f}%")
            r3.metric(ext_nom, f"{p_ext:.1f}%")

            st.divider()
            c_gauche, c_droite = st.columns([2, 1])

            with c_gauche:
                st.subheader("Distribution des Probabilités")
                fig = px.imshow(matrix, 
                                labels=dict(x="Extérieur", y="Domicile", color="Probabilité"),
                                x=[str(i) for i in range(8)], y=[str(i) for i in range(8)],
                                color_continuous_scale='YlOrRd', text_auto='.1%')
                st.plotly_chart(fig, use_container_width=True)
            
            

            with c_droite:
                st.subheader("Scores les plus probables")
                idx_top = np.argsort(matrix.ravel())[-5:][::-1]
                for i in idx_top:
                    h, a = np.unravel_index(i, matrix.shape)
                    st.markdown(f"<div class='score-highlight'><b>{h} - {a}</b> : {matrix[h,a]*100:.1f}%</div>", unsafe_allow_html=True)

                ent = entropy(matrix.ravel())
                conf = max(0, 100 - (ent * 22))
                st.subheader("Indice de Fiabilité")
                st.progress(conf/100)
                st.write(f"Niveau de certitude : {conf:.1f}%")

            st.divider()
            st.subheader("Détails des Puissances Calculées")
            f1, f2, f3 = st.columns(3)
            f1.metric(f"Attaque {dom_nom}", f"{att_final_dom:.2f}")
            f2.metric(f"Attaque {ext_nom}", f"{att_final_ext:.2f}")
            f3.metric("Moyenne Ligue", "1.35")
else:
    st.error("Données indisponibles.")
