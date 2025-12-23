import streamlit as st
import requests
import numpy as np
from scipy.stats import poisson

st.set_page_config(page_title="iTrOz Predictor", layout="wide")

# --- CSS AVEC EFFET AURA ET DISTORSION ---
st.markdown("""
    <style>
    @keyframes subtleDistort {
        0% { transform: scale(1.0); filter: hue-rotate(0deg) brightness(1); }
        50% { transform: scale(1.02) contrast(1.1); filter: hue-rotate(2deg) brightness(1.1); }
        100% { transform: scale(1.0); filter: hue-rotate(0deg) brightness(1); }
    }

    .stApp {
        background-image: url("https://media.giphy.com/media/VZrfUvQjXaGEQy1RSn/giphy.gif");
        background-size: cover;
        background-attachment: fixed;
        animation: subtleDistort 10s infinite ease-in-out;
        overflow: hidden;
    }

    .stApp::before {
        content: "";
        position: fixed;
        top: 0; left: 0; width: 100%; height: 100%;
        background: radial-gradient(circle at var(--mouse-x, 50%) var(--mouse-y, 50%), 
                    rgba(255, 215, 0, 0.15) 0%, 
                    rgba(0,0,0,0) 50%);
        pointer-events: none;
        z-index: 1;
    }

    .stApp > div:first-child { background-color: rgba(0, 0, 0, 0.85); position: relative; z-index: 2; }
    
    h1, h2, h3, p, span, label { color: #FFD700 !important; font-family: 'Monospace', sans-serif; letter-spacing: 2px; }

    div.stButton > button {
        background: rgba(255, 215, 0, 0.03) !important;
        backdrop-filter: blur(25px) !important;
        -webkit-backdrop-filter: blur(25px) !important;
        border: 1px solid rgba(255, 215, 0, 0.2) !important;
        color: #FFD700 !important;
        border-radius: 15px !important;
        height: 70px !important;
        width: 100% !important;
        font-weight: 200 !important;
        text-transform: uppercase !important;
        letter-spacing: 12px !important;
        transition: 0.6s all ease-in-out;
        margin-top: 20px;
    }
    
    div.stButton > button:hover { 
        background: rgba(255, 215, 0, 0.1) !important;
        border: 1px solid rgba(255, 215, 0, 0.6) !important;
        letter-spacing: 16px !important;
        box-shadow: 0 0 40px rgba(255, 215, 0, 0.15);
    }

    div[data-baseweb="select"], div[data-baseweb="input"], .stNumberInput input, .stSelectbox div {
        background-color: rgba(255, 255, 255, 0.05) !important;
        backdrop-filter: blur(12px) !important;
        border: 0.5px solid rgba(255, 215, 0, 0.15) !important;
        border-radius: 10px !important;
        color: #FFD700 !important;
    }

    .verdict-text {
        font-size: 26px; font-weight: 900; text-align: center; padding: 30px;
        letter-spacing: 6px; text-transform: uppercase;
        border-top: 1px solid rgba(255, 215, 0, 0.1);
        border-bottom: 1px solid rgba(255, 215, 0, 0.1);
        margin: 15px 0;
    }

    .bet-card {
        background: rgba(255, 255, 255, 0.02);
        padding: 30px; border-radius: 20px;
        border: 1px solid rgba(255, 215, 0, 0.05);
        margin-bottom: 40px;
    }

    .footer {
        text-align: center; padding: 50px 0 20px 0;
        color: rgba(255, 215, 0, 0.6); font-family: 'Monospace', sans-serif; font-size: 14px;
    }
    .footer a {
        color: #FFD700 !important; text-decoration: none; font-weight: bold;
        border: 1px solid rgba(255, 215, 0, 0.2); padding: 8px 15px; border-radius: 5px;
    }
    </style>

    <script>
    const doc = document.documentElement;
    document.addEventListener('mousemove', e => {
        doc.style.setProperty('--mouse-x', e.clientX + 'px');
        doc.style.setProperty('--mouse-y', e.clientY + 'px');
    });
    </script>
""", unsafe_allow_html=True)

# Configuration API
API_KEY = st.secrets["MY_API_KEY"]
BASE_URL = "https://v3.football.api-sports.io/"
HEADERS = {'x-apisports-key': API_KEY}
SEASON = 2025

@st.cache_data(ttl=3600)
def get_api(endpoint, params):
    try:
        r = requests.get(f"{BASE_URL}{endpoint}", headers=HEADERS, params=params, timeout=12)
        return r.json().get('response', [])
    except: 
        return []

@st.cache_data(ttl=3600)
def get_league_average(league_id, season):
    """Calcule la moyenne réelle de buts de la ligue depuis l'API"""
    standings = get_api("standings", {"league": league_id, "season": season})
    if not standings or not standings[0].get('league', {}).get('standings'):
        return 2.7  # Fallback moyenne Europe
    
    total_goals = 0
    total_matches = 0
    
    for standing in standings[0]['league']['standings'][0]:
        goals_for = standing['all']['goals']['for']
        goals_against = standing['all']['goals']['against']
        played = standing['all']['played']
        
        total_goals += goals_for + goals_against
        total_matches += played
    
    avg = total_goals / total_matches if total_matches > 0 else 2.7
    return avg

@st.cache_data(ttl=3600)
def get_home_away_strength(team_id, league_id, season):
    """Calcule les facteurs de force domicile/extérieur réels de l'équipe"""
    stats = get_api("teams/statistics", {"league": league_id, "season": season, "team": team_id})
    if not stats:
        return 1.0, 1.0
    
    # Buts marqués
    goals_home = float(stats.get('goals',{}).get('for',{}).get('average',{}).get('home') or 0)
    goals_away = float(stats.get('goals',{}).get('for',{}).get('average',{}).get('away') or 0)
    goals_total = float(stats.get('goals',{}).get('for',{}).get('average',{}).get('total') or 1)
    
    # Facteurs relatifs (par rapport à la moyenne de l'équipe)
    home_attack_factor = (goals_home / goals_total) if goals_total > 0 else 1.0
    away_attack_factor = (goals_away / goals_total) if goals_total > 0 else 1.0
    
    # Buts encaissés
    conceded_home = float(stats.get('goals',{}).get('against',{}).get('average',{}).get('home') or 0)
    conceded_away = float(stats.get('goals',{}).get('against',{}).get('average',{}).get('away') or 0)
    conceded_total = float(stats.get('goals',{}).get('against',{}).get('average',{}).get('total') or 1)
    
    home_defense_factor = (conceded_home / conceded_total) if conceded_total > 0 else 1.0
    away_defense_factor = (conceded_away / conceded_total) if conceded_total > 0 else 1.0
    
    return home_attack_factor, away_attack_factor, home_defense_factor, away_defense_factor

@st.cache_data(ttl=1800)
def get_form_factor(team_id, league_id, season):
    """Calcule un facteur de forme basé sur les 5 derniers matchs"""
    fixtures = get_api("fixtures", {"team": team_id, "league": league_id, "season": season, "last": 5})
    if not fixtures:
        return 1.0
    
    points = 0
    matches_count = 0
    goal_diff = 0
    
    for f in fixtures:
        if f['fixture']['status']['short'] != 'FT':
            continue
        
        matches_count += 1
        home_id = f['teams']['home']['id']
        home_goals = f['goals']['home'] or 0
        away_goals = f['goals']['away'] or 0
        
        if home_id == team_id:
            goal_diff += (home_goals - away_goals)
            if home_goals > away_goals: 
                points += 3
            elif home_goals == away_goals: 
                points += 1
        else:
            goal_diff += (away_goals - home_goals)
            if away_goals > home_goals: 
                points += 3
            elif away_goals == home_goals: 
                points += 1
    
    if matches_count == 0:
        return 1.0
    
    # Points par match (moyenne attendue = 1.5)
    points_per_match = points / matches_count
    form_points = 0.85 + (points_per_match - 1.5) / 7.5  # Normalisation autour de 1.0
    
    # Différence de buts (ajustement supplémentaire)
    goal_diff_factor = 1.0 + (goal_diff / matches_count) / 10
    
    # Combiner les deux facteurs
    final_form = form_points * goal_diff_factor
    
    # Limiter entre 0.7 et 1.3 pour éviter les extrêmes
    return max(0.7, min(1.3, final_form))

@st.cache_data(ttl=1800)
def get_head_to_head_factor(team_h_id, team_a_id):
    """Analyse des confrontations directes récentes"""
    h2h = get_api("fixtures/headtohead", {"h2h": f"{team_h_id}-{team_a_id}", "last": 5})
    if not h2h:
        return 1.0, 1.0
    
    goals_h_total = 0
    goals_a_total = 0
    matches_count = 0
    
    for match in h2h:
        if match['fixture']['status']['short'] != 'FT':
            continue
        
        matches_count += 1
        home_id = match['teams']['home']['id']
        home_goals = match['goals']['home'] or 0
        away_goals = match['goals']['away'] or 0
        
        if home_id == team_h_id:
            goals_h_total += home_goals
            goals_a_total += away_goals
        else:
            goals_h_total += away_goals
            goals_a_total += home_goals
    
    if matches_count == 0:
        return 1.0, 1.0
    
    avg_goals_h = goals_h_total / matches_count
    avg_goals_a = goals_a_total / matches_count
    
    # Facteur léger (max +/- 15%)
    h2h_factor_h = 0.925 + (avg_goals_h / 6)
    h2h_factor_a = 0.925 + (avg_goals_a / 6)
    
    return max(0.85, min(1.15, h2h_factor_h)), max(0.85, min(1.15, h2h_factor_a))

if 'simulation_done' not in st.session_state:
    st.session_state.simulation_done = False
    st.session_state.data = {}

st.title("ITROZ PREDICTOR")

leagues = {"La Liga": 140, "Champions League": 2, "Premier League": 39, "Serie A": 135, "Bundesliga": 78, "Ligue 1": 61}
l_name = st.selectbox("CHOISIR LA LIGUE", list(leagues.keys()))
l_id = leagues[l_name]

teams_res = get_api("teams", {"league": l_id, "season": SEASON})
teams = {t['team']['name']: t['team']['id'] for t in teams_res}

if teams:
    sorted_team_names = sorted(teams.keys())
    
    idx_barca = 0
    idx_real = 1
    
    for i, name in enumerate(sorted_team_names):
        if "Barcelona" in name: idx_barca = i
        if "Real Madrid" in name: idx_real = i

    c1, c2 = st.columns(2)
    t_h = c1.selectbox("DOMICILE", sorted_team_names, index=idx_barca)
    t_a = c2.selectbox("EXTÉRIEUR", sorted_team_names, index=idx_real)

    if st.button("Lancer la prédiction"):
        id_h, id_a = teams[t_h], teams[t_a]
        
        with st.spinner("🔍 Analyse approfondie en cours..."):
            # 1. MOYENNE DYNAMIQUE DE LA LIGUE
            league_avg = get_league_average(l_id, SEASON)
            
            # 2. STATISTIQUES DE BASE
            s_h = get_api("teams/statistics", {"league": l_id, "season": SEASON, "team": id_h})
            s_a = get_api("teams/statistics", {"league": l_id, "season": SEASON, "team": id_a})
            
            if s_h and s_a:
                # 3. FACTEURS DOMICILE/EXTÉRIEUR
                h_att_factor, h_away_att, h_def_factor, h_away_def = get_home_away_strength(id_h, l_id, SEASON)
                a_att_factor, a_away_att, a_def_factor, a_away_def = get_home_away_strength(id_a, l_id, SEASON)
                
                # 4. FORME RÉCENTE
                form_h = get_form_factor(id_h, l_id, SEASON)
                form_a = get_form_factor(id_a, l_id, SEASON)
                
                # 5. CONFRONTATIONS DIRECTES
                h2h_h, h2h_a = get_head_to_head_factor(id_h, id_a)
                
                # 6. STATS BRUTES
                att_h_raw = float(s_h.get('goals',{}).get('for',{}).get('average',{}).get('total') or league_avg)
                def_a_raw = float(s_a.get('goals',{}).get('against',{}).get('average',{}).get('total') or league_avg)
                att_a_raw = float(s_a.get('goals',{}).get('for',{}).get('average',{}).get('total') or league_avg)
                def_h_raw = float(s_h.get('goals',{}).get('against',{}).get('average',{}).get('total') or league_avg)
                
                # 7. CALCUL LAMBDA OPTIMISÉ
                # Lambda domicile = (attaque domicile * défense adverse / moyenne ligue) * facteurs
                lh = (att_h_raw * def_a_raw / league_avg) * h_att_factor * form_h * h2h_h
                
                # Lambda extérieur = (attaque extérieur * défense domicile / moyenne ligue) * facteurs
                la = (att_a_raw * def_h_raw / league_avg) * a_away_att * form_a * h2h_a
                
                # 8. MATRICE DYNAMIQUE SELON LES LAMBDAS
                max_goals = int(max(lh, la) * 2.5) + 3
                max_goals = min(max_goals, 12)  # Cap à 12 pour performances
                matrix = np.zeros((max_goals, max_goals))
                
                for x in range(max_goals):
                    for y in range(max_goals): 
                        matrix[x,y] = poisson.pmf(x, lh) * poisson.pmf(y, la)
                
                matrix /= matrix.sum()
                
                st.session_state.data = {
                    'p_h': np.sum(np.tril(matrix, -1)), 
                    'p_n': np.sum(np.diag(matrix)), 
                    'p_a': np.sum(np.triu(matrix, 1)), 
                    'matrix': matrix, 
                    't_h': t_h, 
                    't_a': t_a,
                    'lh': lh,
                    'la': la,
                    'league_avg': league_avg,
                    'form_h': form_h,
                    'form_a': form_a
                }
                st.session_state.simulation_done = True

if st.session_state.simulation_done:
    d = st.session_state.data
    st.write("---")
    
    m1, m2, m3 = st.columns(3)
    m1.metric(d['t_h'], f"{d['p_h']*100:.1f}%")
    m2.metric("NUL", f"{d['p_n']*100:.1f}%")
    m3.metric(d['t_a'], f"{d['p_a']*100:.1f}%")

    st.subheader("🤖 MODE BET")
    st.markdown("<div class='bet-card'>", unsafe_allow_html=True)
    
    b_c1, b_c2, b_c3, b_c4 = st.columns(4)
    bankroll = b_c1.number_input("CAPITAL TOTAL (€)", value=100.0)
    c_h = b_c2.number_input(f"COTE {d['t_h']}", value=2.0)
    c_n = b_c3.number_input("COTE NUL", value=3.0)
    c_a = b_c4.number_input(f"COTE {d['t_a']}", value=3.0)

    dc_1, dc_2, dc_3 = st.columns(3)
    c_hn = dc_1.number_input(f"COTE {d['t_h']} / NUL", value=1.30)
    c_na = dc_2.number_input(f"COTE NUL / {d['t_a']}", value=1.30)
    c_ha = dc_3.number_input(f"COTE {d['t_h']} / {d['t_a']}", value=1.30)

    opts = [
        {"n": d['t_h'], "p": d['p_h'], "c": c_h},
        {"n": "NUL", "p": d['p_n'], "c": c_n},
        {"n": d['t_a'], "p": d['p_a'], "c": c_a},
        {"n": f"{d['t_h']} OU NUL", "p": d['p_h'] + d['p_n'], "c": c_hn},
        {"n": f"NUL OU {d['t_a']}", "p": d['p_n'] + d['p_a'], "c": c_na},
        {"n": f"{d['t_h']} OU {d['t_a']}", "p": d['p_h'] + d['p_a'], "c": c_ha}
    ]

    best_o = max(opts, key=lambda x: x['p'] * x['c'])
    if best_o['p'] * best_o['c'] > 1.02:
        b_val = best_o['c'] - 1
        k_val = ((b_val * best_o['p']) - (1 - best_o['p'])) / b_val if b_val > 0 else 0
        m_finale = bankroll * k_val
        m_finale = max(bankroll * 0.30, m_finale) 
        m_finale = min(m_finale, bankroll * 1.00) 
        
        st.markdown(f"<div class='verdict-text'>IA RECOMMANDE : {best_o['n']} | MISE : {m_finale:.2f}€</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='verdict-text'>AUCUN VALUE DÉTECTÉ</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.subheader("🔍 AUDIT DU TICKET")
    aud1, aud2 = st.columns(2)
    aud_choix = aud1.selectbox("VOTRE PARI", [d['t_h'], "Nul", d['t_a'], f"{d['t_h']} ou Nul", f"Nul ou {d['t_a']}", f"{d['t_h']} ou {d['t_a']}"])
    aud_cote = aud2.number_input("VOTRE COTE", value=1.50)

    p_audit = d['p_h'] if aud_choix == d['t_h'] else (d['p_n'] if aud_choix == "Nul" else d['p_a'])
    if "ou Nul" in aud_choix and d['t_h'] in aud_choix: p_audit = d['p_h'] + d['p_n']
    elif "Nul ou" in aud_choix: p_audit = d['p_n'] + d['p_a']
    elif "ou" in aud_choix: p_audit = d['p_h'] + d['p_a']
    
    audit_val = p_audit * aud_cote
    stat = "SAFE" if audit_val >= 1.10 else ("MID" if audit_val >= 0.98 else "DANGEREUX")
    st.markdown(f"<div class='verdict-text'>AUDIT : {stat} (EV: {audit_val:.2f})</div>", unsafe_allow_html=True)

    st.subheader("SCORES PROBABLES")
    idx = np.unravel_index(np.argsort(d['matrix'].ravel())[-3:][::-1], d['matrix'].shape)
    sc1, sc2, sc3 = st.columns(3)
    for i in range(3):
        with [sc1, sc2, sc3][i]: st.write(f"**{idx[0][i]} - {idx[1][i]}** ({d['matrix'][idx[0][i], idx[1][i]]*100:.1f}%)")
    
    # Infos techniques en expander
    with st.expander("📊 DÉTAILS TECHNIQUES"):
        st.write(f"""
        **Paramètres de calcul :**
        - Moyenne ligue : **{d['league_avg']:.2f}** buts/match
        - Lambda {d['t_h']} : **{d['lh']:.2f}** (forme: {d['form_h']:.2f})
        - Lambda {d['t_a']} : **{d['la']:.2f}** (forme: {d['form_a']:.2f})
        - Matrice : **{d['matrix'].shape[0]}x{d['matrix'].shape[1]}** scores
        """)

st.markdown("""
    <div class='footer'>
        DÉVELOPPÉ PAR ITROZ | 
        <a href='https://github.com/VOTRE_PROFIL' target='_blank'>GITHUB SOURCE</a>
    </div>
""", unsafe_allow_html=True)
