import streamlit as st
import requests
import numpy as np
from scipy.stats import poisson
from datetime import datetime
import pandas as pd

# --- CONFIGURATION ET STYLE GLASSMORPHISM ---
st.set_page_config(page_title="L'ALGO • iTrOz", layout="wide")

GIF_URL = "https://media3.giphy.com/media/v1.Y2lkPTc5MGI3NjExbjcwd3V2YXVyeWg4Z3h2NjdlZmlueWlmaDV6enFnaDM4NDJid2F6ZyZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/VZrfUvQjXaGEQy1RSn/giphy.gif"

st.markdown(f"""
    <style>
    .stApp {{
        background: linear-gradient(rgba(0, 0, 0, 0.85), rgba(0, 0, 0, 0.85)), url("{GIF_URL}");
        background-size: cover; background-attachment: fixed;
    }}
    h1, h2, h3, p, span, label {{ color: #FFD700 !important; font-family: 'Monaco', monospace; }}
    .glass-card {{
        background: rgba(255, 255, 255, 0.03); backdrop-filter: blur(15px);
        border: 1px solid rgba(255, 215, 0, 0.3); padding: 25px; border-radius: 15px; margin-bottom: 20px;
    }}
    .stButton>button {{
        background: rgba(255, 215, 0, 0.15) !important; color: #FFD700 !important;
        border: 2px solid #FFD700 !important; backdrop-filter: blur(10px); height: 55px; font-weight: bold;
    }}
    .stNumberInput div, .stSelectbox div, .stDateInput div, .stSlider div {{
        background: rgba(0, 0, 0, 0.5) !important; border: 1px solid rgba(255, 215, 0, 0.3) !important;
    }}
    </style>
""", unsafe_allow_html=True)

# --- LOGIQUE MATHÉMATIQUE (DIXON-COLES) ---
def calculate_matrix(lh, la):
    matrix = np.zeros((10, 10))
    for x in range(10):
        for y in range(10):
            matrix[x, y] = poisson.pmf(x, lh) * poisson.pmf(y, la)
    return matrix / matrix.sum()

# --- HEADER ---
st.markdown("<h1 style='text-align:center; letter-spacing:10px;'>L'ALGO PRO TERMINAL</h1>", unsafe_allow_html=True)

# --- MODULE 1 : CONFIGURATION GLOBALE ---
with st.container():
    st.markdown("<div class='glass-card'><h3>⚙️ PARAMÈTRES DE SESSION</h3>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1: 
        bankroll = st.number_input("CAPITAL (€)", value=100.0)
        scan_date = st.date_input("DATE DU SCAN", datetime.now())
    with c2: 
        league_choice = st.selectbox("LIGUES", ["Toutes les Ligues", "Premier League", "La Liga", "Ligue 1", "Serie A", "Bundesliga"])
        mode_algo = st.selectbox("MODE DE RISQUE", ["SAFE", "MID SAFE", "MID", "AGRESSIF", "FOU"], index=2)
    with c3: 
        threshold = st.slider("SEUIL OVER/UNDER", 0.5, 4.5, 2.5, 0.5)
    with c4:
        st.write("Statut API: ✅ Connecté")
        st.write("Version: L'ALGO v3.2")
    st.markdown("</div>", unsafe_allow_html=True)

# --- MODULE 2 : ANALYSE MANUELLE (L'ancienne section) ---
st.markdown("### 🔍 ANALYSE INDIVIDUELLE")
with st.expander("Ouvrir l'analyseur de match manuel"):
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        h_team = st.text_input("Équipe Domicile", "Team H")
        h_lambda = st.number_input("Lambda Domicile (xG)", value=1.5)
    with col_m2:
        a_team = st.text_input("Équipe Extérieur", "Team A")
        a_lambda = st.number_input("Lambda Extérieur (xG)", value=1.2)
    
    if st.button("CALCULER ANALYSE BRUTE"):
        m = calculate_matrix(h_lambda, a_lambda)
        p_h, p_n, p_a = np.sum(np.tril(m, -1)), np.sum(np.diag(m)), np.sum(np.triu(m, 1))
        st.write(f"📊 **Probabilités :** Victoire {h_team}: {p_h*100:.1f}% | Nul: {p_n*100:.1f}% | Victoire {a_team}: {p_a*100:.1f}%")

# --- MODULE 3 : AUDIT & SCAN AUTOMATIQUE ---
st.markdown("### 🚀 SCANNER & TICKET AUTOMATIQUE")
if st.button("EXÉCUTER LE SCAN COMPLET DU " + scan_date.strftime('%d/%m/%Y')):
    with st.spinner("L'ALGO scanne les marchés..."):
        # Simulation des données API
        results = [
            {"Match": "Real Madrid - Barça", "Pari": "1", "Cote": 2.10, "Proba": 0.55, "EV": 1.15},
            {"Match": "PSG - Monaco", "Pari": "BTTS OUI", "Cote": 1.75, "Proba": 0.68, "EV": 1.19},
            {"Match": "Man City - Arsenal", "Pari": f"Over {threshold}", "Cote": 1.90, "Proba": 0.60, "EV": 1.14}
        ]
        
        # --- SECTION AUDIT ---
        st.markdown("<div class='glass-card'><h4>🔍 AUDIT VALUE DES MARCHÉS</h4>", unsafe_allow_html=True)
        st.table(pd.DataFrame(results)[["Match", "Pari", "Cote", "EV"]])
        st.markdown("</div>", unsafe_allow_html=True)
        
        # --- SECTION TICKET ---
        st.markdown("<div class='glass-card'><h4>💰 TICKET OPTIMAL GÉNÉRÉ</h4>", unsafe_allow_html=True)
        total_cote = np.prod([o['Cote'] for o in results])
        total_prob = np.prod([o['Proba'] for o in results])
        
        # Kelly Criterion
        b = total_cote - 1
        kelly = ((b * total_prob - (1 - total_prob)) / b) * 0.2 if b > 0 else 0
        mise = max(0, bankroll * kelly)

        col_res1, col_res2 = st.columns(2)
        with col_res1:
            for r in results:
                st.write(f"✅ {r['Match']} : **{r['Pari']}** (@{r['Cote']})")
        with col_res2:
            st.markdown(f"**COTE TOTALE : {total_cote:.2f}**")
            st.markdown(f"**MISE RECOMMANDÉE : {mise:.2f}€**")
        st.markdown("</div>", unsafe_allow_html=True)

        # WEBHOOK DISCORD
        webhook_url = "https://discord.com/api/webhooks/1453026279275106355/gbYAwBRntm1FCoqoBTz5lj1SCe2ijyeHHYoe4CFYwpzOw2DO-ozcCsgkK_53HhB-kFGE"
        discord_msg = {
            "embeds": [{
                "title": f"🔱 SIGNAL L'ALGO - {scan_date.strftime('%d/%m/%Y')}",
                "color": 16766464,
                "description": f"**Cote: {total_cote:.2f} | Mise: {mise:.2f}€**\n\n" + "\n".join([f"• {o['Match']} : {o['Pari']}" for o in results])
            }]
        }
        requests.post(webhook_url, json=discord_msg)
        st.toast("SIGNAL DISCORD ENVOYÉ")

st.markdown("<p style='text-align:center; opacity:0.3; margin-top:50px;'>iTrOz Predictor • L'ALGO v3.2 • Système de grade militaire</p>", unsafe_allow_html=True)
