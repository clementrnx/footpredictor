import streamlit as st
import requests
import numpy as np
from scipy.stats import poisson
from datetime import datetime, timedelta

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
def get_league_context(league_id, season):
    """Récupère le contexte statistique de la ligue"""
    standings = get_api("standings", {"league": league_id, "season": season})
    if not standings or not standings[0].get('league', {}).get('standings'):
        return {'avg_home': 1.5, 'avg_away': 1.2, 'avg_total': 2.7}
    
    total_home_goals = 0
    total_away_goals = 0
    total_home_conceded = 0
    total_away_conceded = 0
    total_matches = 0
    
    for team in standings[0]['league']['standings'][0]:
        home_stats = team['home']
        away_stats = team['away']
        
        total_home_goals += home_stats['goals']['for']
        total_home_conceded += home_stats['goals']['against']
        total_away_goals += away_stats['goals']['for']
        total_away_conceded += away_stats['goals']['against']
        total_matches += home_stats['played']
    
    if total_matches == 0:
        return {'avg_home': 1.5, 'avg_away': 1.2, 'avg_total': 2.7}
    
    return {
        'avg_home': total_home_goals / total_matches,
        'avg_away': total_away_goals / total_matches,
        'avg_home_conceded': total_home_conceded / total_matches,
        'avg_away_conceded': total_away_conceded / total_matches,
        'avg_total': (total_home_goals + total_away_goals) / (total_matches * 2)
    }

@st.cache_data(ttl=1800)
def get_weighted_xg_stats(team_id, league_id, season, is_home=True, use_global=False):
    """
    Calcule les xG moyens pondérés temporellement
    Les matchs récents ont plus de poids (décroissance exponentielle)
    use_global=True : utilise toutes les compétitions
    use_global=False : utilise uniquement la ligue choisie
    """
    if use_global:
        # Toutes compétitions
        fixtures = get_api("fixtures", {"team": team_id, "season": season, "last": 15})
    else:
        # Ligue spécifique
        fixtures = get_api("fixtures", {"team": team_id, "league": league_id, "season": season, "last": 10})
    
    if not fixtures:
        return None
    
    xg_for_weighted = 0
    xg_against_weighted = 0
    goals_for_weighted = 0
    goals_against_weighted = 0
    total_weight = 0
    matches_count = 0
    
    # Tri par date (du plus récent au plus ancien)
    fixtures_sorted = sorted(fixtures, key=lambda x: x['fixture']['date'], reverse=True)
    
    for idx, match in enumerate(fixtures_sorted):
        if match['fixture']['status']['short'] != 'FT':
            continue
        
        # Pondération exponentielle : match le plus récent = poids 1.0, décroissance de 10% par match
        weight = 0.9 ** idx
        
        # Déterminer si l'équipe jouait à domicile ou extérieur
        team_is_home = match['teams']['home']['id'] == team_id
        
        # Filtrer selon le contexte demandé (domicile ou extérieur)
        if is_home and not team_is_home:
            continue
        if not is_home and team_is_home:
            continue
        
        # Récupérer les xG (Expected Goals)
        if team_is_home:
            xg_for = float(match['teams']['home'].get('xg') or match['goals']['home'] or 0)
            xg_against = float(match['teams']['away'].get('xg') or match['goals']['away'] or 0)
            goals_for = match['goals']['home'] or 0
            goals_against = match['goals']['away'] or 0
        else:
            xg_for = float(match['teams']['away'].get('xg') or match['goals']['away'] or 0)
            xg_against = float(match['teams']['home'].get('xg') or match['goals']['home'] or 0)
            goals_for = match['goals']['away'] or 0
            goals_against = match['goals']['home'] or 0
        
        # Accumuler avec pondération
        xg_for_weighted += xg_for * weight
        xg_against_weighted += xg_against * weight
        goals_for_weighted += goals_for * weight
        goals_against_weighted += goals_against * weight
        total_weight += weight
        matches_count += 1
    
    if total_weight == 0 or matches_count == 0:
        return None
    
    return {
        'xg_for': xg_for_weighted / total_weight,
        'xg_against': xg_against_weighted / total_weight,
        'goals_for': goals_for_weighted / total_weight,
        'goals_against': goals_against_weighted / total_weight,
        'matches_count': matches_count
    }

@st.cache_data(ttl=1800)
def get_comprehensive_stats(team_id, league_id, season, use_global=False):
    """Récupère les stats complètes incluant xG pondéré"""
    # Stats globales de l'équipe
    base_stats = get_api("teams/statistics", {"league": league_id, "season": season, "team": team_id})
    
    # xG pondérés
    xg_home = get_weighted_xg_stats(team_id, league_id, season, is_home=True, use_global=use_global)
    xg_away = get_weighted_xg_stats(team_id, league_id, season, is_home=False, use_global=use_global)
    
    return {
        'base': base_stats,
        'xg_home': xg_home,
        'xg_away': xg_away
    }

if 'simulation_done' not in st.session_state:
    st.session_state.simulation_done = False
    st.session_state.data = {}

st.title("ITROZ PREDICTOR")

# Toggle Mode en haut
col_toggle, col_league = st.columns([1, 3])
with col_toggle:
    use_global_stats = st.toggle("📊 MODE GLOBAL", value=False, help="Utilise les stats toutes compétitions au lieu des stats spécifiques à la ligue")

leagues = {"La Liga": 140, "Champions League": 2, "Premier League": 39, "Serie A": 135, "Bundesliga": 78, "Ligue 1": 61}
with col_league:
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
        
        mode_text = "MODE GLOBAL (toutes compétitions)" if use_global_stats else f"MODE CONTEXTE ({l_name})"
        with st.spinner(f"🔍 Analyse xG pondérée + forme récente [{mode_text}]..."):
            # Contexte de la ligue
            league_ctx = get_league_context(l_id, SEASON)
            
            # Stats complètes avec xG pondéré
            stats_h = get_comprehensive_stats(id_h, l_id, SEASON, use_global=use_global_stats)
            stats_a = get_comprehensive_stats(id_a, l_id, SEASON, use_global=use_global_stats)
            
            # Stocker pour debug
            if use_global_stats:
                st.session_state.debug_fixtures_h = get_api("fixtures", {"team": id_h, "season": SEASON, "last": 5})
                st.session_state.debug_fixtures_a = get_api("fixtures", {"team": id_a, "season": SEASON, "last": 5})
            else:
                st.session_state.debug_fixtures_h = get_api("fixtures", {"team": id_h, "league": l_id, "season": SEASON, "last": 5})
                st.session_state.debug_fixtures_a = get_api("fixtures", {"team": id_a, "league": l_id, "season": SEASON, "last": 5})
            
            if stats_h and stats_a:
                s_h = stats_h['base']
                s_a = stats_a['base']
                
                # PRIORITÉ 1 : Utiliser xG pondéré si disponible
                if stats_h['xg_home'] and stats_h['xg_home']['matches_count'] >= 3:
                    att_h_home = stats_h['xg_home']['xg_for']
                    def_h_home = stats_h['xg_home']['xg_against']
                    using_xg_h = True
                else:
                    # Fallback sur buts réels
                    att_h_home = float(s_h.get('goals',{}).get('for',{}).get('average',{}).get('home') or league_ctx['avg_home'])
                    def_h_home = float(s_h.get('goals',{}).get('against',{}).get('average',{}).get('home') or league_ctx['avg_home_conceded'])
                    using_xg_h = False
                
                if stats_a['xg_away'] and stats_a['xg_away']['matches_count'] >= 3:
                    att_a_away = stats_a['xg_away']['xg_for']
                    def_a_away = stats_a['xg_away']['xg_against']
                    using_xg_a = True
                else:
                    # Fallback sur buts réels
                    att_a_away = float(s_a.get('goals',{}).get('for',{}).get('average',{}).get('away') or league_ctx['avg_away'])
                    def_a_away = float(s_a.get('goals',{}).get('against',{}).get('average',{}).get('away') or league_ctx['avg_away_conceded'])
                    using_xg_a = False
                
                # Calcul des forces relatives (modèle Dixon-Coles avec xG)
                attack_strength_h = att_h_home / league_ctx['avg_home'] if league_ctx['avg_home'] > 0 else 1.0
                defense_weakness_a = def_a_away / league_ctx['avg_away_conceded'] if league_ctx['avg_away_conceded'] > 0 else 1.0
                
                attack_strength_a = att_a_away / league_ctx['avg_away'] if league_ctx['avg_away'] > 0 else 1.0
                defense_weakness_h = def_h_home / league_ctx['avg_home_conceded'] if league_ctx['avg_home_conceded'] > 0 else 1.0
                
                # Lambda final avec xG
                lh = league_ctx['avg_home'] * attack_strength_h * defense_weakness_a
                la = league_ctx['avg_away'] * attack_strength_a * defense_weakness_h
                
                # Correction Dixon-Coles pour scores faibles
                tau_00 = -0.13
                tau_10 = 0.065
                tau_01 = 0.065
                tau_11 = 0.13
                
                # Matrice dynamique
                max_goals = int(max(lh, la) * 2.5) + 3
                max_goals = min(max_goals, 10)
                matrix = np.zeros((max_goals, max_goals))
                
                for x in range(max_goals):
                    for y in range(max_goals):
                        prob = poisson.pmf(x, lh) * poisson.pmf(y, la)
                        
                        # Correction Dixon-Coles
                        if x == 0 and y == 0:
                            prob *= (1 + tau_00 * lh * la)
                        elif x == 1 and y == 0:
                            prob *= (1 + tau_10 * lh)
                        elif x == 0 and y == 1:
                            prob *= (1 + tau_01 * la)
                        elif x == 1 and y == 1:
                            prob *= (1 + tau_11)
                        
                        matrix[x, y] = prob
                
                # Normalisation
                matrix = np.maximum(matrix, 0)
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
                    'league_avg': league_ctx['avg_total'],
                    'using_xg_h': using_xg_h,
                    'using_xg_a': using_xg_a,
                    'xg_h_matches': stats_h['xg_home']['matches_count'] if stats_h['xg_home'] else 0,
                    'xg_a_matches': stats_a['xg_away']['matches_count'] if stats_a['xg_away'] else 0,
                    'mode': "Global" if use_global_stats else l_name
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
    idx = np.unravel_index(np.argsort(d['matrix'].ravel())[-5:][::-1], d['matrix'].shape)
    score_cols = st.columns(5)
    for i in range(5):
        with score_cols[i]: 
            st.write(f"**{idx[0][i]} - {idx[1][i]}**")
            st.write(f"{d['matrix'][idx[0][i], idx[1][i]]*100:.1f}%")
    
    with st.expander("🔍 DEBUG API - DERNIERS MATCHS"):
        if 'debug_fixtures_h' in st.session_state:
            st.write(f"**{d['t_h']} - 5 derniers matchs :**")
            for f in st.session_state.debug_fixtures_h[:5]:
                date = f['fixture']['date'][:10]
                home = f['teams']['home']['name']
                away = f['teams']['away']['name']
                score = f"{f['goals']['home']}-{f['goals']['away']}"
                xg_h = f['teams']['home'].get('xg', 'N/A')
                xg_a = f['teams']['away'].get('xg', 'N/A')
                st.write(f"- {date} | {home} vs {away} | {score} | xG: {xg_h}-{xg_a}")
            
            st.write(f"**{d['t_a']} - 5 derniers matchs :**")
            for f in st.session_state.debug_fixtures_a[:5]:
                date = f['fixture']['date'][:10]
                home = f['teams']['home']['name']
                away = f['teams']['away']['name']
                score = f"{f['goals']['home']}-{f['goals']['away']}"
                xg_h = f['teams']['home'].get('xg', 'N/A')
                xg_a = f['teams']['away'].get('xg', 'N/A')
                st.write(f"- {date} | {home} vs {away} | {score} | xG: {xg_h}-{xg_a}")
    
    with st.expander("📊 DÉTAILS TECHNIQUES"):
        xg_status_h = f"✅ xG ({d['xg_h_matches']} matchs)" if d['using_xg_h'] else "⚠️ Buts réels"
        xg_status_a = f"✅ xG ({d['xg_a_matches']} matchs)" if d['using_xg_a'] else "⚠️ Buts réels"
        
        st.write(f"""
        **Modèle Dixon-Coles + xG Pondéré :**
        - **Mode de calcul** : {d['mode']}
        - Moyenne ligue : **{d['league_avg']:.2f}** buts/match
        - λ {d['t_h']} : **{d['lh']:.2f}** buts attendus {xg_status_h}
        - λ {d['t_a']} : **{d['la']:.2f}** buts attendus {xg_status_a}
        - Pondération temporelle : Décroissance exponentielle 10%/match
        - Correction scores faibles : Activée
        - Matrice : **{d['matrix'].shape[0]}x{d['matrix'].shape[1]}** configurations
        
        *{"Stats agrégées toutes compétitions" if d['mode'] == "Global" else f"Stats spécifiques {d['mode']}"}*
        """)


st.markdown("""
    <div class='footer'>
        DÉVELOPPÉ PAR ITROZ | 
        <a href='https://github.com/VOTRE_PROFIL' target='_blank'>GITHUB SOURCE</a>
    </div>
""", unsafe_allow_html=True)
