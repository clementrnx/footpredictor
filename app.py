import streamlit as st
import requests
import numpy as np
from scipy.stats import poisson
from datetime import datetime
import pandas as pd

# --- CONFIGURATION ET STYLE GLASSMORPHISM ---
st.set_page_config(page_title="L'ALGO • TERMINAL", layout="wide")

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
        border: 1px solid rgba(255, 215, 0, 0.3); padding: 25px; border-radius: 15px; margin-bottom: 25px;
    }}
    .stButton>button {{
        background: rgba(255, 215, 0, 0.1) !important; color: #FFD700 !important;
        border: 2px solid #FFD700 !important; backdrop-filter: blur(10px); height: 60px; font-weight: bold; letter-spacing: 3px;
    }}
    .stNumberInput div, .stSelectbox div, .stDateInput div, .stSlider div {{
        background: rgba(255, 255, 255, 0.05) !important; border: 1px solid rgba(255, 215, 0, 0.3) !important;
        backdrop-filter: blur(10px) !important; color: #FFD700 !important;
    }}
    input {{ color: #FFD700 !important; }}
    </style>
""", unsafe_allow_html=True)

# --- LOGIQUE MATHÉMATIQUE ---
def get_ev(proba, cote): return proba * cote

def calculate_kelly(b, p, mode):
    q = 1 - p
    f_star = (b * p - q) / b if b > 0 else 0
    multipliers = {"Kelly Pur (100%)": 1.0, "Demi-Kelly (50%)": 0.5, "Quart-Kelly (25%)": 0.25, "Kelly Prudent (10%)": 0.1}
    return max(0, f_star * multipliers[mode])

# --- HEADER ---
st.markdown("<h1 style='text-align:center; letter-spacing:10px;'>L'ALGO PRO TERMINAL</h1>", unsafe_allow_html=True)

# --- MODULE CONFIGURATION ---
with st.container():
    st.markdown("<div class='glass-card'><h3>⚙️ CONFIGURATION & STRATÉGIE</h3>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        bankroll = st.number_input("CAPITAL TOTAL (€)", value=100.0)
        scan_date = st.date_input("DATE DU SCAN", datetime.now())
    with c2:
        league_choice = st.selectbox("LIGUES", ["Toutes les Ligues", "Premier League", "La Liga", "Ligue 1", "Serie A"])
        kelly_mode = st.selectbox("MODÈLE KELLY CRITERION", ["Kelly Pur (100%)", "Demi-Kelly (50%)", "Quart-Kelly (25%)", "Kelly Prudent (10%)"], index=2)
    with c3:
        threshold = st.slider("SEUIL OVER/UNDER", 0.5, 4.5, 2.5, 0.5)
        algo_risk = st.select_slider("TEMPÉRAMENT ALGO", options=["SAFE", "MID", "FOU"], value="MID")
    st.markdown("</div>", unsafe_allow_html=True)

if st.button("LANCER L'EXÉCUTION DE L'ALGO"):
    with st.spinner("L'ALGO analyse les flux de données..."):
        # Données simulées pour l'exemple
        results = [
            {"Match": "Real Madrid - Barça", "Pari": "Victoire 1", "Cote": 2.15, "Proba": 0.58},
            {"Match": "PSG - Monaco", "Pari": "BTTS OUI", "Cote": 1.70, "Proba": 0.72},
            {"Match": "Liverpool - Arsenal", "Pari": f"Over {threshold}", "Cote": 1.85, "Proba": 0.64}
        ]
        
        for o in results: o['EV'] = get_ev(o['Proba'], o['Cote'])

        # --- MODULE AUDIT ---
        st.markdown("<div class='glass-card'><h3>🔍 AUDIT : ANALYSE DES VALUES</h3>", unsafe_allow_html=True)
        audit_df = pd.DataFrame(results)
        audit_df['Verdict'] = audit_df['EV'].apply(lambda x: "✅ VALUE" if x > 1.05 else "❌ NO VALUE")
        st.table(audit_df[["Match", "Pari", "Cote", "Proba", "EV", "Verdict"]])
        st.markdown("</div>", unsafe_allow_html=True)

        # --- MODULE BET ---
        st.markdown("<div class='glass-card'><h3>💰 BET : TICKET & MISES</h3>", unsafe_allow_html=True)
        
        # Sélection selon le tempérament
        limit = {"SAFE": 1, "MID": 3, "FOU": 5}[algo_risk]
        ticket = sorted(results, key=lambda x: x['EV'], reverse=True)[:limit]
        
        total_cote = np.prod([o['Cote'] for o in ticket])
        total_prob = np.prod([o['Proba'] for o in ticket])
        
        # Calcul de mise avec le Kelly choisi
        b_odds = total_cote - 1
        pct_mise = calculate_kelly(b_odds, total_prob, kelly_mode)
        mise_euros = bankroll * pct_mise

        col_t1, col_t2 = st.columns(2)
        with col_t1:
            st.markdown("#### SÉLECTIONS DU TICKET")
            for t in ticket:
                st.write(f"🔹 {t['Match']} → **{t['Pari']}** (@{t['Cote']})")
        with col_t2:
            st.markdown("#### PERFORMANCE CALCULÉE")
            st.write(f"📈 **Cote Totale :** {total_cote:.2f}")
            st.write(f"🎯 **Proba Combinée :** {total_prob*100:.1f}%")
            st.write(f"⚖️ **Modèle utilisé :** {kelly_mode}")
            st.markdown(f"<h2 style='color:#FFD700;'>MISE : {mise_euros:.2f}€</h2>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # ENVOI DISCORD
        webhook_url = "https://discord.com/api/webhooks/1453026279275106355/gbYAwBRntm1FCoqoBTz5lj1SCe2ijyeHHYoe4CFYwpzOw2DO-ozcCsgkK_53HhB-kFGE"
        embed = {
            "title": f"🔱 TICKET L'ALGO - {kelly_mode}",
            "color": 16766464,
            "description": f"**Cote: {total_cote:.2f} | Mise: {mise_euros:.2f}€**",
            "fields": [{"name": "Matchs", "value": "\n".join([f"• {o['Match']} : {o['Pari']}" for o in ticket])}]
        }
        requests.post(webhook_url, json={"embeds": [embed]})
        st.toast("TRANSMISSION RÉUSSIE")

st.markdown("<p style='text-align:center; opacity:0.3; margin-top:50px;'>L'ALGO v3.5 • QUANTITATIVE BETTING TERMINAL</p>", unsafe_allow_html=True)
