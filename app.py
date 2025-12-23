import streamlit as st
import requests
import numpy as np
from scipy.stats import poisson
from datetime import datetime
import pandas as pd

# --- CONFIGURATION ET STYLE GLASSMORPHISM TOTAL ---
st.set_page_config(page_title="L'ALGO • SYSTEM", layout="wide")

GIF_URL = "https://media3.giphy.com/media/v1.Y2lkPTc5MGI3NjExbjcwd3V2YXVyeWg4Z3h2NjdlZmlueWlmaDV6enFnaDM4NDJid2F6ZyZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/VZrfUvQjXaGEQy1RSn/giphy.gif"

st.markdown(f"""
    <style>
    .stApp {{
        background: linear-gradient(rgba(0, 0, 0, 0.8), rgba(0, 0, 0, 0.8)), url("{GIF_URL}");
        background-size: cover; background-attachment: fixed;
    }}
    h1, h2, h3, p, span, label {{ color: #FFD700 !important; font-family: 'Monaco', monospace; }}
    
    /* GLASSMORPHISM INPUTS */
    .stNumberInput div, .stSelectbox div, .stDateInput div, .stSlider div {{
        background: rgba(255, 255, 255, 0.05) !important;
        backdrop-filter: blur(15px) !important;
        border: 1px solid rgba(255, 215, 0, 0.3) !important;
        border-radius: 10px !important;
    }}
    
    /* BOUTONS ET CARTES */
    .stButton>button {{
        background: rgba(255, 215, 0, 0.1) !important;
        color: #FFD700 !important;
        border: 2px solid #FFD700 !important;
        backdrop-filter: blur(10px);
        letter-spacing: 3px; font-weight: bold; width: 100%; height: 60px;
    }}
    .glass-card {{
        background: rgba(255, 255, 255, 0.02);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 215, 0, 0.2);
        padding: 25px; border-radius: 15px; margin-bottom: 20px;
    }}
    </style>
""", unsafe_allow_html=True)

# --- LOGIQUE ALGO (DIXON-COLES) ---
def get_dc_matrix(lh, la):
    matrix = np.zeros((10, 10))
    for x in range(10):
        for y in range(10):
            matrix[x, y] = poisson.pmf(x, lh) * poisson.pmf(y, la)
    return matrix / matrix.sum()

# --- MODULES DE L'INTERFACE ---
st.markdown("<h1 style='text-align:center; letter-spacing:10px;'>L'ALGO V3.0 PRO</h1>", unsafe_allow_html=True)

# 1. SECTION CONFIGURATION
with st.container():
    st.markdown("<div class='glass-card'><h3>⚙️ CONFIGURATION</h3>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        league_choice = st.selectbox("LIGUES", ["Toutes les Ligues", "Premier League", "La Liga", "Ligue 1", "Serie A", "Bundesliga"])
        leagues_dict = {"Premier League": 39, "La Liga": 140, "Ligue 1": 61, "Serie A": 135, "Bundesliga": 78}
    with c2: bankroll = st.number_input("CAPITAL (€)", value=100.0)
    with c3: mode_algo = st.select_slider("TEMPÉRAMENT", options=["SAFE", "MID SAFE", "MID", "MID AGRESSIF", "AGRESSIF", "FOU"], value="MID")
    with c4: threshold = st.slider("SEUIL O/U", 0.5, 4.5, 2.5, 0.5)
    st.markdown("</div>", unsafe_allow_html=True)

if st.button("LANCER L'EXÉCUTION DE L'ALGO"):
    with st.spinner("SCAN MULTI-MARCHÉS EN COURS..."):
        # Logique de récupération API (Simulée pour l'exemple)
        # En production, boucler sur leagues_dict si "Toutes les Ligues"
        results = [
            {"Match": "Real Madrid - Barça", "Pari": "1", "Cote": 2.10, "Proba": 0.55, "EV": 1.15},
            {"Match": "PSG - Monaco", "Pari": "BTTS OUI", "Cote": 1.75, "Proba": 0.68, "EV": 1.19},
            {"Match": "Man City - Arsenal", "Pari": f"Over {threshold}", "Cote": 1.90, "Proba": 0.60, "EV": 1.14},
            {"Match": "Inter - Juve", "Pari": "N2", "Cote": 1.65, "Proba": 0.70, "EV": 1.15}
        ]
        
        # 2. SECTION ANALYSE (Probabilités pures)
        st.markdown("<div class='glass-card'><h3>📊 ANALYSE PRÉDICTIVE</h3>", unsafe_allow_html=True)
        st.table(pd.DataFrame(results)[["Match", "Proba"]].rename(columns={"Proba": "Confiance ALGO"}))
        st.markdown("</div>", unsafe_allow_html=True)

        # 3. SECTION AUDIT (Calcul des Values)
        st.markdown("<div class='glass-card'><h3>🔍 AUDIT DES COTES (VALUE)</h3>", unsafe_allow_html=True)
        st.table(pd.DataFrame(results)[["Match", "Pari", "Cote", "EV"]])
        st.markdown("</div>", unsafe_allow_html=True)

        # 4. SECTION BET (Le Ticket Final)
        st.markdown("<div class='glass-card'><h3>💰 BET : TICKET OPTIMAL</h3>", unsafe_allow_html=True)
        
        # Sélection du meilleur combiné selon l'EV
        best_bets = sorted(results, key=lambda x: x['EV'], reverse=True)[:3]
        total_cote = np.prod([o['Cote'] for o in best_bets])
        total_prob = np.prod([o['Proba'] for o in best_bets])
        
        # Kelly Fractionnée
        b = total_cote - 1
        kelly = ((b * total_prob - (1 - total_prob)) / b) * 0.2 if b > 0 else 0
        mise = max(0, bankroll * kelly)

        col_a, col_b = st.columns(2)
        with col_a:
            for b in best_bets:
                st.write(f"✅ {b['Match']} → **{b['Pari']}** (@{b['Cote']})")
        with col_b:
            st.markdown(f"#### COTE TOTALE : {total_cote:.2f}")
            st.markdown(f"#### MISE CONSEILLÉE : {mise:.2f}€")
        st.markdown("</div>", unsafe_allow_html=True)

        # ENVOI DISCORD
        webhook_data = {
            "embeds": [
                {
                    "title": f"🔱 SIGNAL L'ALGO - MODE {mode_algo}",
                    "color": 16766464,
                    "fields": [
                        {"name": "TICKET", "value": "\n".join([f"{o['Match']} : {o['Pari']}" for o in best_bets])},
                        {"name": "STATS", "value": f"Cote: {total_cote:.2f} | Mise: {mise:.2f}€ | EV: {(total_cote*total_prob):.2f}"}
                    ]
                }
            ]
        }
        requests.post("https://discord.com/api/webhooks/1453026279275106355/gbYAwBRntm1FCoqoBTz5lj1SCe2ijyeHHYoe4CFYwpzOw2DO-ozcCsgkK_53HhB-kFGE", json=webhook_data)
        st.toast("SIGNAL TRANSMIS SUR DISCORD")

st.markdown("<p style='text-align:center; opacity:0.2;'>iTrOz Predictor • Full Algo Terminal • 2025</p>", unsafe_allow_html=True)
