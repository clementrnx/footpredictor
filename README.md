# footpredictor
🏆 iTrOz Predictor | Absolute Edition
iTrOz Predictor est un moteur d'arbitrage statistique conçu pour modéliser les probabilités de résultats des rencontres de football. Contrairement aux approches basées sur l'intuition, cet outil repose exclusivement sur trois piliers mathématiques avancés pour quantifier l'incertitude et la performance.

🧠 Architecture Mathématique
Le projet utilise des modèles de probabilités discrètes et des théories de l'information pour transformer des données historiques en indicateurs décisionnels.

1. Distribution de Poisson Croisée

Le moteur de calcul de buts repose sur la Loi de Poisson. Elle est utilisée pour modéliser le nombre d'événements (buts) se produisant dans un intervalle de temps fixe.

P(k;λ)= 
k!
λ 
k
 e 
−λ
 
​	
 
Dans iTrOz Predictor, nous calculons deux variables indépendantes λ 
H
​	
  (domicile) et λ 
A
​	
 (extérieur) en croisant :

Le potentiel offensif de l'équipe A.

La vulnérabilité défensive de l'équipe B.

Un coefficient de pondération pour l'avantage du terrain.

La confrontation est ensuite modélisée par une Distribution de Skellam, qui calcule la probabilité de la différence de buts entre les deux distributions de Poisson, permettant d'extraire les probabilités de Victoire, Nul et Défaite.

2. Entropie de Shannon (Indice de Chaos)

Pour évaluer la fiabilité d'une prédiction, l'outil intègre le concept d'Entropie de Shannon. Elle mesure le degré de désordre ou d'incertitude contenu dans les probabilités calculées.

H(X)=− 
i=1
∑
n
​	
 P(x 
i
​	
 )log 
2
​	
 P(x 
i
​	
 )
Entropie Faible : Les probabilités sont concentrées sur une issue. Le match est structurellement "lisible" et l'ordre statistique domine.

Entropie Élevée : Les probabilités sont équilibrées (33%/33%/33%). Le système est en état de chaos maximal. Dans ce cas, l'outil signale que l'aléa sportif (chance, erreurs d'arbitrage) prendra le pas sur la logique des chiffres.

3. Pondération Temporelle Exponentielle (Boost de Forme)

Le modèle intègre un Recency Bias Control (biais de récence). Les statistiques de l'ensemble de la saison sont ajustées par un multiplicateur dynamique basé sur les 5 derniers matchs.

Ce calcul permet de corriger l'inertie des moyennes classiques : une équipe en crise de résultats verra ses probabilités de succès dégradées exponentiellement, même si son historique de début de saison était excellent. Cela permet de coller à la "dynamique de forme" réelle du vestiaire.

🛠️ Stack Technique
Calculs : NumPy / SciPy (Algèbres linéaires et distributions)

Interface : Streamlit (Visualisation de données)

Flux : API REST (Football-Data.org)

Développeur : itrozola

GitHub : clementrnx
