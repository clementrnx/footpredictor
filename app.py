import streamlit as st
import requests
import numpy as np
from scipy.stats import poisson
from datetime import datetime

# --- CONFIGURATION ET STYLE GLASSMORPHISM PRO ---
st.set_page_config(page_title="L'ALGO • iTrOz", layout="wide")

GIF_URL = "https://media3.giphy.com/media/v1.Y2lkPTc5MGI3NjExbjcwd3V2YXVyeWg4Z3h2NjdlZmlueWlmaDV6enFnaDM4NDJid2F6ZyZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/VZrfUvQjXaGEQy1RSn/giphy.gif"

st.markdown(f"""
    <style>
    /* 1. FOND ET OVERLAY */
    .stApp {{
        background: linear-gradient(rgba(0, 0, 0, 0.75), rgba(0, 0, 0, 0.75)), url("{GIF_URL}");
        background-size: cover; background-attachment: fixed;
    }}

    /* 2. TEXTE ET TITRES */
    h1, h2, h3, p, span, label {{ 
        color: #FFD700 !important; 
        font-family: 'Monaco', monospace; 
        text-shadow: 2px 2px 4px rgba(0,0,0,0.9);
    }}

    /* 3. INPUTS GLASSMORPHISM (Sélecteurs, Nombres, Dates) */
    .stNumberInput div, .stSelectbox div, .stDateInput div, div[data-baseweb="select"], .stSlider div {{
        background-color: rgba(255, 255, 255, 0.05) !important;
        backdrop-filter: blur(12px) !important;
        -webkit-backdrop-filter: blur(12px) !important;
        border: 1px solid rgba(255, 215, 0, 0.3) !important;
        border_radius: 10px !important;
        color: #FFD700 !important;
    }}
    
    /* Cible spécifiquement l'intérieur des inputs */
    input {{
        color: #FFD700 !important;
        background: transparent !important;
        border: none !important;
    }}

    /* 4. BOUTONS GLASSMORPHISM */
    .stButton>button {{
        width: 100%;
        background: rgba(255, 215, 0, 0.1) !important;
        color: #FFD700 !important;
        border: 2px solid rgba(255, 215, 0, 0.4) !important;
        border-radius: 15px !important;
        height: 65px;
        font-weight: bold;
        letter-spacing: 5px;
        backdrop-filter: blur(15px);
        transition: all 0.4s ease;
    }}
    .stButton>button:hover {{
        background: rgba(255, 215, 0, 0.3) !important;
        border: 2px solid #FFD700 !important;
        box-shadow: 0 0 25px rgba(255, 215, 0, 0.4);
        transform: scale(1.01);
    }}

    /* 5. CARTES DE RÉSULTATS (BET-CARD) */
    .bet-card {{
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 215, 0, 0.2);
        border-radius: 20px;
        padding: 25px;
        margin-top: 20px;
        box-shadow: 0 10px 40px rgba(0,0,0,0.5);
    }}
    </style>
""", unsafe_allow_html=True)

# --- CONFIG API & WEBHOOK ---
API_KEY = st.secrets["MY_API_KEY"]
HEADERS = {'x-apisports-key': API_KEY}
DISCORD_WEBHOOK = "https://discord.com/api/webhooks/1453026279275106355/gbYAwBRntm1FCoqoBTz5lj1SCe2ijyeHHYoe4CFYwpzOw2DO-ozcCsgkK_53HhB-kFGE"

ALGO_MODES = {
    "SAFE": {"min_ev": 1.15, "kelly": 0.10, "max_legs": 2},
    "MID SAFE": {"min_ev": 1.10, "kelly": 0.20, "max_legs": 3},
    "MID": {"min_ev": 1.07, "kelly": 0.35, "max_legs": 4},
    "MID AGRESSIF": {"min_ev": 1.04, "kelly": 0.50, "max_legs": 5},
    "AGRESSIF": {"min_ev": 1.02, "kelly": 0.75, "max_legs": 8},
    "FOU": {"min_ev": 0.98, "kelly": 1.00, "max_legs": 15}
}

# --- FONCTION MATHS ---
def get_dc_probs(lh, la):
    matrix = np.zeros((8, 8))
    for x in range(8):
        for y in range(8):
            matrix[x, y] = poisson.pmf(x, lh) * poisson.pmf(y, la)
    matrix /= matrix.sum()
    return matrix

# --- INTERFACE ---
st.markdown("<h1 style='text-align:center; letter-spacing:15px; margin-bottom:40px;'>L'ALGO</h1>", unsafe_allow_html=True)

# Conteneur des Inputs
with st.container():
    st.markdown("<div class='bet-card'>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1: bankroll = st.number_input("CAPITAL TOTAL (€)", value=100.0)
    with c2: mode_name = st.selectbox("TEMPÉRAMENT", list(ALGO_MODES.keys()), index=2)
    with c3: threshold = st.slider("SEUIL O/U", 0.5, 4.5, 2.5, 1.0)
    with c4: scan_date = st.date_input("DATE ANALYSE", datetime.now())
    st.markdown("</div>", unsafe_allow_html=True)

if st.button("EXÉCUTER LE SCAN"):
    with st.spinner("L'ALGO interroge les serveurs..."):
        # Logique de scan simplifiée (Dixon-Coles + Cotes API)
        # On simule ici la récupération pour le rendu visuel
        valid_selections = [
            {"Match": "Paris SG - Marseille", "Pari": "Victoire Home", "Cote": 1.65, "Proba": 0.72, "EV": 1.18},
            {"Match": "Lyon - Monaco", "Pari": f"Over {threshold}", "Cote": 1.80, "Proba": 0.65, "EV": 1.17}
        ]
        
        conf = ALGO_MODES[mode_name]
        ticket = valid_selections[:conf['max_legs']]
        
        total_cote = np.prod([o['Cote'] for o in ticket])
        total_prob = np.prod([o['Proba'] for o in ticket])
        mise_f = ( (total_cote * total_prob - (1 - total_prob)) / total_cote ) * conf['kelly'] * bankroll

        # Affichage du Ticket
        st.markdown(f"""
            <div class='bet-card' style='text-align:center;'>
                <h3>📑 TICKET OPTIMAL DÉTECTÉ</h3>
                <p style='font-size:32px;'>COTE TOTALE : <b>{total_cote:.2f}</b></p>
                <p style='font-size:24px; color:white !important;'>MISE CONSEILLÉE : <b>{max(0, mise_f):.2f}€</b></p>
                <hr style='border: 1px solid rgba(255,215,0,0.2);'>
            </div>
        """, unsafe_allow_html=True)
        st.table(ticket)
        
        # Envoi Discord
        embed = {
            "title": f"🔱 SIGNAL L'ALGO - {mode_name}",
            "color": 16766464,
            "description": f"**Cote: {total_cote:.2f} | Mise: {max(0, mise_f):.2f}€**",
            "fields": [{"name": "Matchs", "value": "\n".join([f"{o['Match']} : {o['Pari']}" for o in ticket])}]
        }
        requests.post(DISCORD_WEBHOOK, json={"embeds": [embed]})
        st.toast("SIGNAL ENVOYÉ")

st.markdown("<p style='text-align:center; margin-top:50px; opacity:0.3;'>TERMINAL SÉCURISÉ L'ALGO v3.0</p>", unsafe_allow_html=True)
