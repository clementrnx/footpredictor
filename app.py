import streamlit as st
import requests
import numpy as np
from scipy.stats import poisson

# ================== PAGE & CSS ==================
st.set_page_config(page_title="iTrOz Predictor", layout="wide")

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

.score-card {
    background: rgba(255, 255, 255, 0.02);
    padding: 20px; border-radius: 15px;
    border: 1px solid rgba(255, 215, 0, 0.05);
    margin-top: 20px;
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

# ================== CONFIG ==================
API_KEY = st.secrets["MY_API_KEY"]
BASE_URL = "https://v3.football.api-sports.io/"
SEASON = 2025
MAX_STAKE = 0.70

# ================== API ==================
@st.cache_data(ttl=3600)
def get_api(endpoint, params):
    try:
        r = requests.get(f"{BASE_URL}{endpoint}", headers={'x-apisports-key': API_KEY}, params=params, timeout=10)
        return r.json().get('response', [])
    except:
        return []

@st.cache_data(ttl=3600)
def get_league_context(league_id, season):
    standings = get_api("standings", {"league": league_id, "season": season})
    if not standings:
        return {'avg_home': 1.5, 'avg_away': 1.2, 'avg_home_conceded': 1.2, 'avg_away_conceded': 1.5}

    th, ta, th_c, ta_c, m = 0, 0, 0, 0, 0
    for t in standings[0]['league']['standings'][0]:
        th += t['home']['goals']['for']
        th_c += t['home']['goals']['against']
        ta += t['away']['goals']['for']
        ta_c += t['away']['goals']['against']
        m += t['home']['played']

    return {
        'avg_home': th / m,
        'avg_away': ta / m,
        'avg_home_conceded': th_c / m,
        'avg_away_conceded': ta_c / m
    }

# ================== BET LOGIC ==================
def kelly_fraction(p, c):
    b = c - 1
    return max(((b * p) - (1 - p)) / b, 0) if b > 0 else 0

def model_confidence(d):
    score = 0
    score += 1 if d['using_xg_h'] else 0
    score += 1 if d['using_xg_a'] else 0
    score += 1 if abs(d['lh'] - d['la']) > 0.6 else 0
    return score  # 0 à 3

def mode_conservative(signals, bankroll):
    v = [s for s in signals if s['ev'] >= 1.05 and s['confidence'] >= 2]
    if not v: return None
    b = max(v, key=lambda x: x['ev'])
    return b, bankroll * min(b['kelly'] * 0.35, MAX_STAKE)

def mode_balanced(signals, bankroll):
    v = [s for s in signals if s['ev'] >= 1.02 and s['confidence'] >= 1]
    if not v: return None
    b = max(v, key=lambda x: x['ev'])
    return b, bankroll * min(b['kelly'] * 0.70, MAX_STAKE)

def mode_aggressive(signals, bankroll):
    v = [s for s in signals if s['ev'] >= 1.00]
    if not v: return None
    b = max(v, key=lambda x: x['ev'])
    return b, bankroll * min(b['kelly'], MAX_STAKE)

def mode_value(signals, bankroll):
    v = [s for s in signals if s['ev'] > 1.00]
    if not v: return None
    b = max(v, key=lambda x: x['ev'])
    return b, bankroll * min(0.10, MAX_STAKE)

# ================== UI ==================
st.title("ITROZ PREDICTOR")

leagues = {"La Liga": 140, "Premier League": 39, "Serie A": 135}
league = st.selectbox("LIGUE", leagues.keys())
league_id = leagues[league]

teams_res = get_api("teams", {"league": league_id, "season": SEASON})
teams = {t['team']['name']: t['team']['id'] for t in teams_res}

c1, c2 = st.columns(2)
home = c1.selectbox("DOMICILE", teams.keys())
away = c2.selectbox("EXTÉRIEUR", teams.keys())

if st.button("Lancer la prédiction"):
    ctx = get_league_context(league_id, SEASON)

    lh = ctx['avg_home']
    la = ctx['avg_away']

    # ---------------- SCORES -----------------
    max_g = 10
    matrix = np.zeros((max_g, max_g))
    for i in range(max_g):
        for j in range(max_g):
            matrix[i, j] = poisson.pmf(i, lh) * poisson.pmf(j, la)
    matrix /= matrix.sum()

    p_h = np.sum(np.tril(matrix, -1))
    p_n = np.sum(np.diag(matrix))
    p_a = np.sum(np.triu(matrix, 1))

    st.metric(home, f"{p_h*100:.1f}%")
    st.metric("NUL", f"{p_n*100:.1f}%")
    st.metric(away, f"{p_a*100:.1f}%")

    # ---------------- PROBABILITÉ DE SCORE -----------------
    st.subheader("Probabilité de score exact")
    score_html = "<div class='score-card'><ul>"
    for i in range(5):
        for j in range(5):
            prob = matrix[i,j]*100
            if prob > 2:  # n'affiche que >2%
                score_html += f"<li>{home} {i} - {j} {away} : {prob:.1f}%</li>"
    score_html += "</ul></div>"
    st.markdown(score_html, unsafe_allow_html=True)

    # ---------------- MODE BET -----------------
    st.subheader("MODE BET")

    bankroll = st.number_input("CAPITAL (€)", value=100.0)
    c_h = st.number_input(f"COTE {home}", value=2.0)
    c_n = st.number_input("COTE NUL", value=3.0)
    c_a = st.number_input(f"COTE {away}", value=3.0)

    bet_mode = st.selectbox(
        "MODE DE BET",
        ["CONSERVATIVE", "BALANCED", "AGGRESSIVE", "PURE VALUE"]
    )

    opts = [
        {"n": home, "p": p_h, "c": c_h},
        {"n": "NUL", "p": p_n, "c": c_n},
        {"n": away, "p": p_a, "c": c_a}
    ]

    d = {
        "lh": lh, "la": la,
        "using_xg_h": True,
        "using_xg_a": True
    }

    conf = model_confidence(d)

    signals = []
    for o in opts:
        signals.append({
            "name": o['n'],
            "p": o['p'],
            "c": o['c'],
            "ev": o['p'] * o['c'],
            "kelly": kelly_fraction(o['p'], o['c']),
            "confidence": conf
        })

    if bet_mode == "CONSERVATIVE":
        res = mode_conservative(signals, bankroll)
    elif bet_mode == "BALANCED":
        res = mode_balanced(signals, bankroll)
    elif bet_mode == "AGGRESSIVE":
        res = mode_aggressive(signals, bankroll)
    else:
        res = mode_value(signals, bankroll)

    if res:
        o, stake = res
        st.markdown(
            f"<div class='verdict-text'>IA RECOMMANDE : {o['name']} | MISE : {stake:.2f}€</div>",
            unsafe_allow_html=True
        )

        # ---------- MINI AUDIT ----------
        audit_html = f"""
        <div class='score-card'>
        <strong>Audit du ticket :</strong><br>
        EV : {o['ev']:.2f} | Kelly : {o['kelly']:.2f} | Confiance : {o['confidence']}/3<br>
        Mise recommandée : {stake:.2f}€
        </div>
        """
        st.markdown(audit_html, unsafe_allow_html=True)
    else:
        st.markdown("<div class='verdict-text'>AUCUN VALUE DÉTECTÉ</div>", unsafe_allow_html=True)

st.markdown("""
<div class='footer'>
DÉVELOPPÉ PAR ITROZ | 
<a href='https://github.com/VOTRE_PROFIL' target='_blank'>GITHUB SOURCE</a>
</div>
""", unsafe_allow_html=True)
