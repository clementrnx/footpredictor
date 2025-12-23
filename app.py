import streamlit as st
import requests
import numpy as np
from scipy.stats import poisson
from datetime import datetime
import pandas as pd

# --- CONFIGURATION ---
st.set_page_config(page_title="Clementrnxx Predictor V5.5 - Élite", layout="wide")

# Paramètres de la Philosophie "Rentabilité Maximale"
# On ne limite pas les matchs, on laisse le Score Élite filtrer.
MIN_RELIABILITY_SCORE = 0.90 # Seuil de coupure pour l'entrée dans le ticket

API_KEY = st.secrets["MY_API_KEY"]
HEADERS = {'x-apisports-key': API_KEY}
BASE_URL = "https://v3.football.api-sports.io/"
LEAGUES = {"La Liga": 140, "Premier League": 39, "Champions League": 2, "Ligue 1": 61, "Serie A": 135, "Bundesliga": 78}

def get_team_stats(team_id, league_id):
    params = {"team": team_id, "season": 2025, "last": 15}
    r = requests.get(f"{BASE_URL}fixtures", headers=HEADERS, params=params).json().get('response', [])
    if not r: return 1.3, 1.3
    scored = [m['goals']['home'] if m['teams']['home']['id'] == team_id else m['goals']['away'] for m in r if m['goals']['home'] is not None]
    conceded = [m['goals']['away'] if m['teams']['home']['id'] == team_id else m['goals']['home'] for m in r if m['goals']['home'] is not None]
    w = [0.96 ** i for i in range(len(scored))]
    return sum(s * i for s, i in zip(scored, w)) / sum(w), sum(c * i for c, i in zip(conceded, w)) / sum(w)

def get_probs(lh, la):
    matrix = np.zeros((12, 12))
    for x in range(12):
        for y in range(12):
            matrix[x, y] = poisson.pmf(x, lh) * poisson.pmf(y, la)
    matrix /= matrix.sum()
    return {"H": np.sum(np.tril(matrix, -1)), "D": np.sum(np.diag(matrix)), "A": np.sum(np.triu(matrix, 1))}

st.title("💎 SCANNER ÉLITE : MAXIMISATION EV × P")
st.markdown("---")

bankroll = st.sidebar.number_input("Ta Bankroll (€)", value=100.0)
date_sel = st.sidebar.date_input("Date des matchs", datetime.now())

if st.button("GÉNÉRER LE TICKET MATHÉMATIQUEMENT SUPÉRIEUR"):
    results = []
    with st.spinner("Calcul de la rentabilité réelle..."):
        for l_name, l_id in LEAGUES.items():
            fixtures = requests.get(f"{BASE_URL}fixtures", headers=HEADERS, params={"league": l_id, "season": 2025, "date": date_sel.strftime('%Y-%m-%d')}).json().get('response', [])
            
            for f in fixtures:
                if f['fixture']['status']['short'] != "NS": continue
                ah, dh = get_team_stats(f['teams']['home']['id'], l_id)
                aa, da = get_team_stats(f['teams']['away']['id'], l_id)
                lh, la = (ah * da) ** 0.5 * 1.05, (aa * dh) ** 0.5 * 0.95
                pr = get_probs(lh, la)
                
                odds_res = requests.get(f"{BASE_URL}odds", headers=HEADERS, params={"fixture": f['fixture']['id']}).json().get('response', [])
                if odds_res and odds_res[0]['bookmakers']:
                    for bet in odds_res[0]['bookmakers'][0]['bets']:
                        if bet['name'] == "Match Winner":
                            for v in bet['values']:
                                p = pr['H'] if v['value'] == 'Home' else pr['D'] if v['value'] == 'Draw' else pr['A']
                                cote = float(v['odd'])
                                ev = p * cote
                                # SCORE ÉLITE = EV * P (soit P^2 * Cote)
                                elite_score = ev * p 
                                
                                if elite_score >= MIN_RELIABILITY_SCORE:
                                    results.append({
                                        "Match": f"{f['teams']['home']['name']} - {f['teams']['away']['name']}",
                                        "Pari": v['value'],
                                        "Cote": cote,
                                        "Proba": f"{p:.1%}",
                                        "EV": round(ev, 2),
                                        "Score_Elite": round(elite_score, 3),
                                        "raw_p": p
                                    })

    if results:
        # On trie pour avoir le match le plus "mathématiquement parfait" en premier
        df = pd.DataFrame(results).sort_values(by="Score_Elite", ascending=False)
        
        total_odd = np.prod(df['Cote'].values)
        total_proba = np.prod(df['raw_p'].values)
        
        st.success(f"Ticket généré avec {len(df)} matchs optimisés.")
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Cote Totale", f"@{total_odd:.2f}")
        c2.metric("Fiabilité Combinée", f"{total_proba:.2%}")
        c3.metric("Mise conseillée (Kelly)", f"{bankroll * 0.05:.2f}€")

        st.table(df.drop(columns=['raw_p']))
    else:
        st.warning("Aucun match ne présente un score d'élite suffisant aujourd'hui.")
