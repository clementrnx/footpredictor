# --- LOGIQUE OPTIMISEUR HYBRIDE V5.5 ---
if st.button("GÉNÉRER ", key="btn_gen"):
    # ... (garder ton début de code pour la récupération des fixtures) ...
    
    all_market_opportunities = []

    for f in fixtures:
        # 1. Calcul des probabilités de base
        att_h, def_h = get_team_stats(f['teams']['home']['id'], lid, scope_scan=="OVER-ALL")
        att_a, def_a = get_team_stats(f['teams']['away']['id'], lid, scope_scan=="OVER-ALL")
        lh, la = (att_h * def_a) ** 0.5 * 1.08, (att_a * def_h) ** 0.5 * 0.92
        pr = calculate_perfect_probs(lh, la)
        
        # 2. Mapping complet des marchés pour l'optimisation
        # Format: (Nom Marché API, Valeur API, Label UI, Probabilité)
        market_map = [
            ("Match Winner", "Home", f['teams']['home']['name'], pr['p_h']),
            ("Match Winner", "Away", f['teams']['away']['name'], pr['p_a']),
            ("Double Chance", "Home/Draw", f"{f['teams']['home']['name']}/N", pr['p_1n']),
            ("Double Chance", "Draw/Away", f"N/{f['teams']['away']['name']}", pr['p_n2']),
            ("Both Teams Score", "Yes", "BTTS OUI", pr['p_btts']),
            ("Both Teams Score", "No", "BTTS NON", pr['p_nobtts'])
        ]

        # 3. Récupération des cotes et filtrage par efficience
        odds_data = get_api("odds", {"fixture": f['fixture']['id']})
        if odds_data:
            bookies = odds_data[0].get('bookmakers', [])
            if bookies:
                main_bookie = bookies[0] # On prend le plus complet
                for b_n, b_v, label, prob in market_map:
                    # On ne garde que ce qui passe ton filtre de risque initial
                    if prob >= risk_cfg['p']:
                        for bet in main_bookie.get('bets', []):
                            if bet['name'] == b_n:
                                for val in bet['values']:
                                    if val['value'] == b_v:
                                        cote = float(val['odd'])
                                        ev = prob * cote
                                        if ev >= risk_cfg['ev']:
                                            all_market_opportunities.append({
                                                "MATCH": f"{f['teams']['home']['name']} - {f['teams']['away']['name']}",
                                                "PARI": label,
                                                "COTE": cote,
                                                "PROBA": prob,
                                                "EV": ev
                                            })

    # --- PHASE D'OPTIMISATION DU TICKET PARFAIT ---
    # On trie par Probabilité * EV pour avoir le meilleur compromis
    all_market_opportunities = sorted(all_market_opportunities, key=lambda x: x['PROBA'] * x['EV'], reverse=True)
    
    # On limite le nombre de matchs selon ton curseur
    final_ticket = all_market_opportunities[:max_legs]
    
    if final_ticket:
        total_odd = np.prod([x['COTE'] for x in final_ticket])
        # Probabilité de vie finale du ticket (Produit des probas)
        pdv_finale = np.prod([x['PROBA'] for x in final_ticket])
        
        mise_totale = bank_scan * risk_cfg['kelly']
        
        st.markdown(f"""
            <div class='verdict-box'>
                <h2 style='color:#FFD700;'>🚀 TICKET OPTIMISÉ</h2>
                <p>COTE TOTALE : <b>@{total_odd:.2f}</b></p>
                <p>PROBABILITÉ DE VIE FINALE : <b>{pdv_finale*100:.1f}%</b></p>
                <p>MISE CONSEILLÉE : <b>{mise_totale:.2f}€</b></p>
            </div>
        """, unsafe_allow_html=True)
        
        st.table(pdv_ticket_df) # Affichage des détails
