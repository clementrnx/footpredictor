import streamlit as st
import requests
import numpy as np
from scipy.stats import poisson
from datetime import datetime

# --- CONFIGURATION ET STYLE ---
st.set_page_config(page_title="iTrOz Predictor Pro", layout="wide")

st.markdown("""
    <style>
    @keyframes subtleDistort {
        0% { transform: scale(1.0); filter: hue-rotate(0deg) brightness(1); }
        50% { transform: scale(1.01) contrast(1.1); filter: hue-rotate(2deg) brightness(1.1); }
        100% { transform: scale(1.0); filter: hue-rotate(0deg) brightness(1); }
    }

    .stApp {
        background-image: url("https://media.giphy.com/media/VZrfUvQjXaGEQy1RSn/giphy.gif");
        background-size: cover;
        background-attachment: fixed;
        animation: subtleDistort 10s infinite ease-in-out;
    }

    .stApp > div:first-child { background-color: rgba(0, 0, 0, 0.88); }
    
    h1, h2, h3, p, span, label { color: #FFD700 !important; font-family: 'Monospace', sans-serif; letter-spacing: 2px; }

    /* Boutons de Mode */
    .stButton > button {
        background: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid rgba(255, 215, 0, 0.2) !important;
        color: #FFD700 !important;
        transition: 0.3s;
    }
    
    .stButton > button:hover {
        border-color: #FFD700 !important;
        box-shadow: 0 0 15px rgba(255, 215, 0, 0.3);
    }

    .verdict-card {
        background: rgba(255, 215, 0, 0.05);
        border-left: 5px solid #FFD700;
        padding: 25px;
        border-radius: 10px;
        text-align: center;
        margin: 20px 0;
    }

    .badge {
        padding: 5px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: bold;
        text-transform: uppercase;
        margin-bottom: 10px;
        display: inline-block;
    }
    </style>
""", unsafe_allow_html=True)

# --- CONFIGURATION API ---
API_KEY = st.secrets["MY_API_KEY"]
BASE_URL = "https://v3.football.api-sports.io/"
HEADERS = {'x-apisports-key': API_KEY}
SEASON = 2025

# --- FONCTIONS TECHNIQUES ---
@st.cache_data(ttl=3600)
def get_api(endpoint, params):
    try:
        r = requests.get(f"{BASE_URL}{endpoint}", headers=HEADERS, params=params, timeout=12)
        return r.json().get('response', [])
    except: return []

@st.cache_data(ttl=3600)
def get_league_context(league_id, season):
    standings = get_api("standings", {"league": league_id, "season": season})
    if not standings: return {'avg_home': 1.5, 'avg_away': 1.2, 'avg_home_conceded': 1.2, 'avg_away_conceded': 1.5, 'avg_total': 2.7}
    
    # Calcul simplifié des moyennes de ligue
    return {'avg_home': 1.56, 'avg_away': 1.23, 'avg_home_conceded': 1.23, 'avg_away_conceded': 1.56, 'avg_total': 2.79}

@st.cache_data(ttl=1800)
def get_weighted_stats(team_id, league_id, season, is_home=True, use_global=False):
    params = {"team": team_id, "season": season, "last": 15} if use_global else {"team": team_id, "league": league_id, "season": season, "last": 10}
    fixtures = get_api("fixtures", params)
    if not fixtures: return None
    
    total_w, xg_f, xg_a, goals_f, goals_a = 0, 0, 0, 0, 0
    for idx, m in enumerate(fixtures):
        if m['fixture']['status']['short'] != 'FT': continue
        weight = 0.9 ** idx
        home = m['teams']['home']['id'] == team_id
        if is_home and not home: continue
        if not is_home and home: continue
        
        f_val = float(m['teams']['home' if home else 'away'].get('xg') or m['goals']['home' if home else 'away'] or 0)
        a_val = float(m['teams']['away' if home else 'home'].get('xg') or m['goals']['away' if home else 'home'] or 0)
        
        xg_f += f_val * weight
        xg_a += a_val * weight
        total_w += weight
        
    return {'xg_for': xg_f/total_w, 'xg_against': xg_a/total_w, 'count': len(fixtures)} if total_w > 0 else None

# --- INTERFACE PRINCIPALE ---
st.title("ITROZ PREDICTOR PRO")

# Sidebar - Mode Global
with st.sidebar:
    st.header("PARAMÈTRES")
    use_global_stats = st.toggle("📊 MODE GLOBAL (Toutes compétitions)", value=False)
    st.write("---")
    st.info("Le Mode Global est conseillé pour les coupes ou les débuts de saison.")

leagues = {"La Liga": 140, "Champions League": 2, "Premier League": 39, "Serie A": 135, "Bundesliga": 78, "Ligue 1": 61}
l_name = st.selectbox("LIGUE", list(leagues.keys()))
l_id = leagues[l_name]

teams_res = get_api("teams", {"league": l_id, "season": SEASON})
teams = {t['team']['name']: t['team']['id'] for t in teams_res}

if teams:
    c1, c2 = st.columns(2)
    t_h = c1.selectbox("DOMICILE", sorted(teams.keys()), index=0)
    t_a = c2.selectbox("EXTÉRIEUR", sorted(teams.keys()), index=1)

    # --- CALCULS ---
    if st.button("LANCER L'ANALYSE XG"):
        with st.spinner("Calcul des probabilités Dixon-Coles..."):
            ctx = get_league_context(l_id, SEASON)
            s_h = get_weighted_stats(teams[t_h], l_id, SEASON, True, use_global_stats)
            s_a = get_weighted_stats(teams[t_a], l_id, SEASON, False, use_global_stats)

            if s_h and s_a:
                lh = ctx['avg_home'] * (s_h['xg_for'] / ctx['avg_home']) * (s_a['xg_against'] / ctx['avg_home_conceded'])
                la = ctx['avg_away'] * (s_a['xg_for'] / ctx['avg_away']) * (s_h['xg_against'] / ctx['avg_away_conceded'])
                
                # Matrice de Poisson
                matrix = np.zeros((7, 7))
                for x in range(7):
                    for y in range(7):
                        prob = poisson.pmf(x, lh) * poisson.pmf(y, la)
                        # Correction Dixon-Coles simplifiée
                        if x==0 and y==0: prob *= 0.87
                        if (x==1 and y==0) or (x==0 and y==1): prob *= 1.05
                        matrix[x, y] = prob
                matrix /= matrix.sum()

                st.session_state.data = {
                    'p_h': np.sum(np.tril(matrix, -1)), 'p_n': np.sum(np.diag(matrix)), 'p_a': np.sum(np.triu(matrix, 1)),
                    'matrix': matrix, 't_h': t_h, 't_a': t_a, 'lh': lh, 'la': la
                }
                st.session_state.done = True

# --- SECTION BETTING ---
if st.session_state.get('done'):
    d = st.session_state.data
    st.write("---")
    
    # Sélecteur de profil "Cool"
    st.subheader("⚡ SÉLECTION DU PROFIL DE JEU")
    if 'risk_mode' not in st.session_state: st.session_state.risk_mode = "SAFE"
    
    m_col1, m_col2, m_col3 = st.columns(3)
    with m_col1:
        if st.button("🛡️ SAFE"): st.session_state.risk_mode = "SAFE"
    with m_col2:
        if st.button("⚖️ MID"): st.session_state.risk_mode = "MID"
    with m_col3:
        if st.button("🔥 JOUEUR"): st.session_state.risk_mode = "JOUEUR"

    # Configuration des modes
    conf_map = {
        "SAFE":   {"seuil": 1.05, "kelly": 0.25, "max": 0.05, "color": "#00FFCC", "badge": "Bouclier Actif"},
        "MID":    {"seuil": 1.02, "kelly": 0.50, "max": 0.15, "color": "#FFD700", "badge": "Optimisé"},
        "JOUEUR": {"seuil": 1.001, "kelly": 1.0, "max": 0.40, "color": "#FF3131", "badge": "Déglinguage / Volume"}
    }
    c = conf_map[st.session_state.risk_mode]

    # Input Cotes
    bc1, bc2, bc3, bc4 = st.columns(4)
    bankroll = bc1.number_input("Capital (€)", value=100.0)
    c_h = bc2.number_input(f"Cote {d['t_h']}", value=2.0)
    c_n = bc3.number_input("Cote Nul", value=3.2)
    c_a = bc4.number_input(f"Cote {d['t_a']}", value=3.5)

    # Calcul du pari
    opts = [
        {"n": d['t_h'], "p": d['p_h'], "c": c_h},
        {"n": "Match Nul", "p": d['p_n'], "c": c_n},
        {"n": d['t_a'], "p": d['p_a'], "c": c_a}
    ]
    
    # Filtrage par seuil (Mode JOUEUR accepte tout avantage > 0.1%)
    valides = [o for o in opts if (o['p'] * o['c']) >= c['seuil']]
    
    if valides:
        best = max(valides, key=lambda x: x['p'] * x['c'])
        edge = (best['p'] * best['c']) - 1
        f_kelly = (edge / (best['c'] - 1)) if best['c'] > 1 else 0
        mise = min(bankroll * f_kelly * c['kelly'], bankroll * c['max'])

        if mise > 0.1:
            st.markdown(f"""
                <div class='verdict-card' style='border-color: {c['color']};'>
                    <div class='badge' style='background: {c['color']}; color: black;'>{c['badge']}</div>
                    <h2 style='margin:0; color: white !important;'>CONSEIL : {best['n']}</h2>
                    <h1 style='font-size: 45px; color: {c['color']} !important;'>{mise:.2f} €</h1>
                    <p style='color: #aaa !important;'>Edge détecté : +{edge*100:.1f}% | Mode : {st.session_state.risk_mode}</p>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.warning("L'avantage est présent mais la mise recommandée est trop faible pour être affichée.")
    else:
        st.markdown(f"<div class='verdict-text'>AUCUNE VALUE (Seuil {st.session_state.risk_mode}: {c['seuil']})</div>", unsafe_allow_html=True)

    # Scores Exacts
    st.write("### 🎯 SCORES LES PLUS PROBABLES")
    idx = np.unravel_index(np.argsort(d['matrix'].ravel())[-5:][::-1], d['matrix'].shape)
    cols = st.columns(5)
    for i in range(5):
        cols[i].metric(f"{idx[0][i]} - {idx[1][i]}", f"{d['matrix'][idx[0][i], idx[1][i]]*100:.1f}%")

st.markdown("<div style='text-align:center; margin-top:50px; opacity:0.5;'>iTrOz Predictor v2.5 | 2025 Model</div>", unsafe_allow_html=True)
