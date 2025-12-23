with tab2:
    st.subheader(" GÉNÉRATEUR DE TICKETS OPTIMISÉ")
    gc1, gc2, gc3, gc4 = st.columns(4)
    l_scan = gc1.selectbox("CHAMPIONNAT", ["TOUTES LES LEAGUES"] + list(LEAGUES_DICT.keys()), key="l_scan")
    d_range = gc2.date_input("PÉRIODE DU SCAN", [datetime.now(), datetime.now()], key="d_scan_range")
    bank_scan = gc3.number_input("FOND DISPONIBLE (€) ", value=100.0, key="b_scan_input")
    
    # --- NOUVEAU CURSEUR NOMBRE DE MATCHS ---
    max_legs = gc4.slider("NB MATCHS MAX DANS LE TICKET", 1, 30, 3)
    
    scope_scan = st.select_slider("DATA SCAN", options=["LEAGUE ONLY", "OVER-ALL"], value="OVER-ALL", key="scope_scan")
    selected_markets = st.multiselect("MARCHÉS À ANALYSER", ["ISSUE SIMPLE", "DOUBLE CHANCE", "BTTS (OUI/NON)"], default=["ISSUE SIMPLE", "DOUBLE CHANCE"], key="markets_scan")
    
    risk_mode = st.select_slider("RISQUE", options=["SAFE", "MID-SAFE", "MID", "MID-AGGRESSIF", "AGGRESSIF"], value="MID", key="risk_scan")
    risk_cfg = RISK_LEVELS[risk_mode]
    
    if st.button("GÉNÉRER LE TICKET PARFAIT", key="btn_gen"):
        if isinstance(d_range, (list, tuple)) and len(d_range) == 2:
            start_date, end_date = d_range
            date_list = pd.date_range(start=start_date, end=end_date).tolist()
        else:
            st.error("⚠️ Sélectionnez une période complète."); st.stop()

        lids = LEAGUES_DICT.values() if l_scan == "TOUTES LES LEAGUES" else [LEAGUES_DICT[l_scan]]
        all_opps = []
        progress_bar = st.progress(0)
        
        # 1. SCAN MULTI-MARCHÉS
        for idx_d, current_date in enumerate(date_list):
            date_str = current_date.strftime('%Y-%m-%d')
            for lid in lids:
                fixtures = get_api("fixtures", {"league": lid, "season": SEASON, "date": date_str})
                for f in fixtures:
                    if f['fixture']['status']['short'] != "NS": continue
                    
                    # Calcul Probas Poisson
                    att_h, def_h = get_team_stats(f['teams']['home']['id'], lid, scope_scan=="OVER-ALL")
                    att_a, def_a = get_team_stats(f['teams']['away']['id'], lid, scope_scan=="OVER-ALL")
                    lh, la = (att_h * def_a) ** 0.5 * 1.08, (att_a * def_h) ** 0.5 * 0.92
                    pr = calculate_perfect_probs(lh, la)
                    h_n, a_n = f['teams']['home']['name'], f['teams']['away']['name']

                    # Mapping des opportunités sur tous les marchés
                    tests = []
                    if "ISSUE SIMPLE" in selected_markets:
                        tests += [(h_n, pr['p_h'], "Match Winner", "Home"), (a_n, pr['p_a'], "Match Winner", "Away")]
                    if "DOUBLE CHANCE" in selected_markets:
                        tests += [(f"{h_n}/N", pr['p_1n'], "Double Chance", "Home/Draw"), (f"N/{a_n}", pr['p_n2'], "Double Chance", "Draw/Away")]
                    if "BTTS (OUI/NON)" in selected_markets:
                        tests += [("BTTS OUI", pr['p_btts'], "Both Teams Score", "Yes"), ("BTTS NON", pr['p_nobtts'], "Both Teams Score", "No")]

                    # Vérification des cotes pour chaque opportunité
                    if tests:
                        odds_data = get_api("odds", {"fixture": f['fixture']['id']})
                        if odds_data:
                            for b_label, p, m_name, m_val in tests:
                                if p >= risk_cfg['p']: # Filtre de probabilité minimal
                                    for bet in odds_data[0]['bookmakers'][0]['bets']:
                                        if bet['name'] == m_name:
                                            for o in bet['values']:
                                                if o['value'] == m_val:
                                                    cote = float(o['odd'])
                                                    if (p * cote) >= risk_cfg['ev']: # Filtre Value
                                                        all_opps.append({
                                                            "MATCH": f"{h_n} - {a_n}",
                                                            "PARI": b_label,
                                                            "COTE": cote,
                                                            "PROBA": p,
                                                            "SCORE_VIE": p * (p * cote) # Optimisation Probabilité x Value
                                                        })
            progress_bar.progress((idx_d + 1) / len(date_list))

        # 2. CONSTRUCTION DU TICKET PARFAIT (Tri par Score de Vie)
        # On évite de prendre plusieurs fois le même match pour diversifier le risque
        unique_matches = []
        final_selection = []
        all_opps = sorted(all_opps, key=lambda x: x['SCORE_VIE'], reverse=True)

        for opp in all_opps:
            if opp['MATCH'] not in unique_matches and len(final_selection) < max_legs:
                final_selection.append(opp)
                unique_matches.append(opp['MATCH'])

        # 3. AFFICHAGE ET CALCULS FINAUX
        if final_selection:
            total_odd = np.prod([x['COTE'] for x in final_selection])
            prob_vie_finale = np.prod([x['PROBA'] for x in final_selection])
            mise_sugg = bank_scan * risk_cfg['kelly']
            
            st.markdown(f"""
                <div class='verdict-box'>
                    <h2 style='color:#FFD700;'>🔥 TICKET PARFAIT GÉNÉRÉ</h2>
                    <p style='font-size:1.2rem;'>Cote Totale : <b>@{total_odd:.2f}</b> | Probabilité de Vie : <b>{prob_vie_finale*100:.1f}%</b></p>
                    <p>Mise suggérée : <b>{mise_sugg:.2f}€</b> (Kelly Mode {risk_mode})</p>
                </div>
            """, unsafe_allow_html=True)
            
            st.table(pd.DataFrame(final_selection)[["MATCH", "PARI", "COTE", "PROBA"]])
            send_to_discord(final_selection, total_odd, risk_mode)
        else:
            st.error("Aucune opportunité ne respecte vos critères de risque et de probabilité.")
