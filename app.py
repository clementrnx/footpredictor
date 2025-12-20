import streamlit as st
import requests
import math
import numpy as np

st.set_page_config(page_title="iTrOz Predictor | Absolute", layout="wide")

st.markdown("""
    <style>
    .stApp { background: #050505; color: #FFFFFF; }
    h1, .stMetricValue { color: #FFD700 !important; font-weight: 900 !important; }
    .stSelectbox label { color: #FFD700 !important; }
    .side-link {
        border: 1px solid #FFD700; color: #FFD700 !important;
        padding: 10px; border-radius: 5px; text-decoration: none; 
        display: block; text-align: center; margin-bottom: 10px;
        font-weight: bold;
    }
    .side-link:hover { background: #FFD700; color: #000 !important; }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def get_absolute_data():
    token = "f30f164b1e1b47cfb7ede5d459f5ab54"
    headers = {'X-Auth-Token': token}
    teams = {}
    for league in ["PD", "PL", "FL1"]:
        try:
            r = requests.get(f"https://api.football-data.org/v4/competitions/{league}/standings", headers=headers)
            rm = requests.get(f"https://api.football-data.org/v4/competitions/{league}/matches?status=FINISHED", headers=headers)
            matches = rm.json()['matches']
            for team in r.json()['standings'][0]['table']:
                t_id = team['team']['id']
                name = team['team']['name']
                pts_forme, count = 0, 0
                for m in reversed(matches):
                    if count >= 5: break
                    if m['homeTeam']['id'] == t_id or m['awayTeam']['id'] == t_id:
                        win = (m['homeTeam']['id'] == t_id and m['score']['winner'] == 'HOME_TEAM') or \
                              (m['awayTeam']['id'] == t_id and m['score']['winner'] == 'AWAY_TEAM')
                        draw = m['score']['winner'] == 'DRAW'
                        pts_forme += 3 if win else 1 if draw else 0
                        count += 1
                form_boost = 0.85 + (pts_forme / 15) * 0.30
                teams[name] = {
                    'gf': team['goalsFor'], 'ga': team['goalsAgainst'],
                    'played': team['playedGames'], 'logo': team['team']['crest'],
                    'form_boost': form_boost
                }
        except: continue
    return teams

st.title("🏆 iTrOz Predictor")
data = get_absolute_data()

if data:
    t_list = sorted(list(data.keys()))
    col_a, col_b = st.columns(2)
    with col_a:
        home = st.selectbox("DOMICILE", t_list, index=t_list.index("FC Barcelona") if "FC Barcelona" in t_list else 0)
        st.image(data[home]['logo'], width=100)
    with col_b:
        away = st.selectbox("EXTÉRIEUR", t_list, index=t_list.index("Real Madrid CF") if "Real Madrid CF" in t_list else 1)
        st.image(data[away]['logo'], width=100)

    if st.button("START"):
        h, a = data[home], data[away]
        l_h = (h['gf']/h['played']) * (a['ga']/a['played']) * h['form_boost'] * 1.25
        l_a = (a['gf']/a['played']) * (h['ga']/h['played']) * a['form_boost'] * 0.85
        p_h, p_a, p_d, total = 0, 0, 0, 0
        for i in range(10):
            for j in range(10):
                prob = ((math.exp(-l_h)*l_h**i)/math.factorial(i)) * \
                       ((math.exp(-l_a)*l_a**j)/math.factorial(j))
                total += prob
                if i > j: p_h += prob
                elif j > i: p_a += prob
                else: p_d += prob
        p_h, p_d, p_a = p_h/total, p_d/total, p_a/total
        entropy = -sum(p * math.log2(p) for p in [p_h, p_d, p_a] if p > 0)
        chaos_score = (entropy / 1.58) * 100
        st.markdown("---")
        m1, m2, m3 = st.columns(3)
        m1.metric("VICTOIRE DOM", f"{p_h*100:.1f}%")
        m2.metric("NUL", f"{p_d*100:.1f}%")
        m3.metric("VICTOIRE EXT", f"{p_a*100:.1f}%")
        st.subheader(f"Indice de Chaos : {chaos_score:.1f}%")
        st.progress(chaos_score/100)
        if chaos_score > 80: st.error("MATCH IMPRÉVISIBLE")
        elif chaos_score < 60: st.success("MATCH LISIBLE")
        else: st.warning("MATCH VOLATIL")

st.sidebar.markdown(f"""
    <br><br>
    <a href="https://github.com/clementrnx/" target="_blank" class="side-link">📂 GitHub : clementrnx</a>
    <a href="https://discord.com/users/itrozola" target="_blank" class="side-link">👾 Discord : itrozola</a>
    """, unsafe_allow_html=True)
