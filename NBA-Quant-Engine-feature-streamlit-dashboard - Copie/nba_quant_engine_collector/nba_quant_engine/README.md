# NBA Quant Engine — Streamlit + DunkAPI collector

Application Streamlit et collecteur Python pour importer les équipes NBA, calendriers/résultats, box-scores et cotes DunkAPI dans le schéma PostgreSQL Supabase nba.

## Interface Streamlit

L'application propose deux vues :

- **Vue d'ensemble** : volumes d'équipes, matchs, joueurs, statistiques et snapshots de cotes ; derniers matchs ; qualité temporelle PIT ; historique des imports.
- **Collecteur** : lancement manuel des imports avec sélection de la saison, du type de saison, du statut des matchs, du nombre maximal de matchs et du plafond de crédits.

L'ouverture de l'application ne déclenche aucune collecte. Les imports utilisent le même NBACollector que la CLI, conservent les upserts idempotents, l'historique d'ingestion, les payloads bruts et les contrôles de provenance temporelle.

## Déploiement sur Streamlit Community Cloud

1. Ouvrir Streamlit Community Cloud et choisir **New app**.
2. Sélectionner ce dépôt GitHub : azmourad067-stack/NBA-Quant-Engine.
3. Choisir la branche feature/streamlit-dashboard.
4. Définir le chemin du fichier principal :
   streamlit_app.py (à la racine du dépôt)
5. Dans les options avancées, sélectionner Python 3.12.
6. Dans **App → Settings → Secrets**, ajouter les variables suivantes :

    APP_ACCESS_PASSWORD = "un-mot-de-passe-long-et-unique"
    DUNKAPI_API_KEY = "YOUR_DUNKAPI_API_KEY"
    SUPABASE_DB_URL = "postgresql://postgres...."
    NBA_DB_SCHEMA = "nba"
    DUNKAPI_BASE_URL = "https://mcp.dunkapi.com"
    MAX_CREDITS_PER_RUN = "350"
    REQUEST_TIMEOUT_SECONDS = "60"
    REQUEST_INTERVAL_SECONDS = "1.05"

SUPABASE_DB_URL doit être l'URI de connexion PostgreSQL obtenue dans Supabase → **Connect** → **Session Pooler**. Ce n'est pas la clé API service_role. APP_ACCESS_PASSWORD est obligatoire : l'application reste bloquée s'il n'est pas défini. Choisissez un mot de passe long et unique. Ne mettez jamais les secrets dans un fichier suivi par Git ou dans le code source.

Le fichier requirements.txt est situé dans le même dossier que streamlit_app.py, afin que Streamlit Cloud installe les dépendances de l'application.

Pour les URI Supabase utilisant le pooler transactionnel, l'application retire le paramètre Prisma `pgbouncer=true`. Elle écarte également `hostaddr` lorsqu'il contient un nom DNS (ce paramètre doit contenir une IP littérale) et laisse PostgreSQL résoudre le nom porté par `host`. Les autres options, notamment `sslmode=require`, sont conservées. Psycopg utilise `prepare_threshold=None` pour éviter les requêtes préparées sur le pooler.

Pour les connexions via le pooler transactionnel Supabase, l'application supprime le paramètre `pgbouncer=true` (spécifique à Prisma) et configure Psycopg avec `prepare_threshold=None`. Conservez les autres paramètres PostgreSQL de l'URI, notamment `sslmode=require`.


## Lancement local

Depuis le dossier nba_quant_engine_collector/nba_quant_engine :

    python -m venv .venv
    .\.venv\Scripts\Activate.ps1
    pip install -r requirements.txt
    Copy-Item .env.example .env
    streamlit run streamlit_app.py

Renseigner DUNKAPI_API_KEY et SUPABASE_DB_URL dans .env avant de lancer l'application. Sous macOS/Linux, activer l'environnement avec source .venv/bin/activate et copier le modèle avec cp .env.example .env.

La CLI reste disponible. Pour lancer ou reprendre le backfill des saisons régulières 2023–2024 à 2025–2026 avec le plafond de 120 crédits :

    python -m collector.cli --season-start 2024 --season-end 2026 --season-type regular --status final --max-games 5000 --max-credits 120 --skip-odds

Relancer exactement la même commande jusqu'à ce que le rapport indique que toutes les saisons sont parcourues et qu'il ne reste plus de matchs terminés à enrichir. Le collecteur reprend les offsets de pagination dans `nba_ingestion_runs`, et interroge le backlog en base pour détailler les matchs les plus anciens incomplets. Les box-scores complets ne sont pas redemandés. Le plafond est appliqué à chaque lancement, pas à l'ensemble des lancements.

## Import et intégrité des données

Tables alimentées :

- nba_teams, nba_games
- nba_team_game_stats, nba_players, nba_player_game_stats
- nba_odds_snapshots
- nba_ingestion_runs, nba_raw_payloads pour l'audit des imports

Le collecteur vérifie le schéma existant avant l'écriture. Il ne crée, ne supprime, ne tronque et ne modifie aucune table. Les upserts s'appuient sur les identifiants du fournisseur et préservent les valeurs non nulles déjà enregistrées.

Règles de données :

- Les statistiques non retournées par le fournisseur (par exemple pace, possessions ou ratings par 100 possessions) restent NULL.
- Les heures réelles de tip-off ne sont pas inventées.
- Une cote capturée avant le tip-off peut être verified_point_in_time.
- Une cote capturée après le tip-off est postgame_backfill.
- Une cote sans horodatage source fiable reste unknown.
- Les cotes historiques récupérées après le match ne doivent pas être utilisées comme prix pré-match dans les backtests.
- Les opening/closing moneyline sont importées lorsqu'elles sont fournies. Les lignes et prix spread/total non disponibles ne sont pas reconstruits artificiellement.
- Les réponses API sont conservées dans nba_raw_payloads et les appels sont consignés dans nba_ingestion_runs.

Le plafond de crédits de la page de collecte est une limite locale estimative : le fournisseur reste la référence pour la consommation réelle. Pour le backfill historique, le plafond est de 120 crédits par lancement et la pagination peut parcourir jusqu'à 5 000 matchs par lancement ; la limite de crédits, et non ce plafond de sécurité, arrête généralement le lot.

## Tests

    pytest -q

Les tests existants couvrent les mappings DunkAPI → Supabase, les conversions de cotes américaines, les box-scores et la classification PIT. Ils n'appellent ni DunkAPI ni Supabase.

## Couverture fournisseur

La documentation DunkAPI décrit des matchs et box-scores historiques depuis 2003–04, mais l'historique des cotes commence en 2013–14 avec des périodes et marchés incomplets. Les rapports de blessures ne peuvent être collectés qu'à partir du lancement de leur collecte. Vérifier le catalogue de données du fournisseur avant de planifier un import historique massif : https://dunkapi.com/data.


## Phase B — enrichissement des matchs terminés

La préconfiguration du collecteur passe en phase B :

- saison régulière 2026–2027 ;
- statut `final` ;
- jusqu'à 25 matchs ;
- box-scores activés ;
- cotes désactivées pour ne pas consommer de crédits sur des dates où DunkAPI ne renvoie aucune ligne ;
- plafond local de 120 crédits.

L'enrichissement est incrémental : avant chaque collecte de box-scores, le collecteur vérifie les matchs qui n'ont pas encore de lignes dans les deux tables `nba_team_game_stats` et `nba_player_game_stats`. Les matchs déjà complets sont ignorés. Un match partiellement enrichi est repris. Les upserts restent idempotents.

Lors de la collecte du 17 novembre 2026, DunkAPI a renvoyé les calendriers des 21–24 novembre mais des tableaux `books` vides pour chaque rencontre, avec la note fournisseur `No sportsbook lines for this day`. Ce n'est pas une erreur de parsing : aucune cote n'est disponible dans la réponse. Les cotes restent donc désactivées dans la phase B ; elles pourront être réactivées lors d'une collecte ultérieure si le fournisseur publie des lignes.

## Phase C — backfill historique paginé et reprenable

La préconfiguration du collecteur cible maintenant les saisons régulières 2023–2024 à 2025–2026, statut `final`, jusqu'à 5 000 matchs parcourus par lancement et un plafond de 120 crédits.

- La pagination `list_games` reprend au dernier `next_offset` réussi, stocké dans les métadonnées de `nba_ingestion_runs` pour chaque saison, type de saison et filtre de statut.
- Les offsets ne sont avancés qu'après une page réussie. Une relance ne recommence donc pas inutilement à la première page.
- Le collecteur réserve 25 crédits avant de poursuivre la pagination quand les box-scores sont activés.
- Les matchs terminés nécessitant encore des statistiques sont recherchés dans toute la plage de saisons en base, et enrichis par ordre chronologique avec les crédits restants.
- L'enrichissement est considéré complet lorsqu'il existe au moins deux lignes de statistiques équipes et au moins une ligne de statistiques joueurs.
- Les cotes sont désactivées dans cette étape : elles feront l'objet d'une collecte distincte, avec leur propre suivi de progression et leur coût. Les lignes horodatées après le tip-off ou sans heure source fiable ne sont jamais qualifiées de PIT.

Le fournisseur documente `limit`, `offset` et `next_offset` pour `list_games`. Le paramètre saison utilise l'année de fin (2024 = 2023–2024). La reprise s'appuie sur ces curseurs et sur les upserts idempotents.

## Moteur prédictif pré-match v1.0

Le dossier `engine/` contient un moteur complet de prédiction des matchs NBA, en lecture seule (aucun appel DunkAPI, aucune écriture en base).

### Ce qu'il prédit

Pour chaque match à venir (`game_status = 'scheduled'`) :

- probabilité de victoire domicile / extérieur et cotes « justes » (1/p) ;
- marge de points attendue (donc un spread modèle) et total de points attendu ;
- value bet éventuel : si une cote moneyline `verified_point_in_time` est disponible et que l'espérance de gain dépasse 3 %, le côté, la cote, l'EV et une mise en quart de Kelly (plafonnée à 2 % de la bankroll).

### Architecture

| Module | Rôle |
| --- | --- |
| `engine/elo.py` | Elo avec marge de victoire, avantage du terrain (100 pts) et régression de 25 % entre saisons. |
| `engine/dataset.py` | Variables pré-match chronologiques : Elo, repos, back-to-back, matchs sur 7 jours, forme 5/10 matchs, bilan de saison, moyennes exponentielles des ratings offensif/défensif/net pour 100 possessions, rythme, *four factors* (eFG %, TOV %, ORB %, FTr) et confrontations attaque vs défense adverse. Agrégation des cotes PIT (moneyline sans marge, spread, total). |
| `engine/models.py` | `PredictionEngine` : régression logistique régularisée, gradient boosting (HistGradientBoosting), régression ridge de la marge (convertie en probabilité via une loi normale), ridge du total, et ensemble moyen (logistique + marge ; le gradient boosting est calculé et affiché mais exclu de l'ensemble, car il est le moins bon hors échantillon). Indicateur d'accord entre modèles (Fort / Moyen / Faible). |
| `engine/evaluation.py` | Backtest walk-forward à fenêtre croissante : réentraînement tous les N matchs, chaque match test prédit par un modèle entraîné uniquement sur les jours précédents. Métriques : précision, Brier, log-loss, AUC, calibration, détail par saison, MAE marge/total, taux contre le spread et over/under quand les lignes existent. |
| `engine/betting.py` | EV, Kelly fractionnel, simulation chronologique de bankroll sur les prédictions hors échantillon, et évaluation de cotes saisies à la main. |
| `engine/sources.py` | Chargement depuis Supabase, depuis des CSV au format des tables, depuis le fichier public FiveThirtyEight `nbaallelo.csv`, ou génération d'une ligue synthétique pour la démo et les tests. |
| `engine/service.py` | Orchestration (rapport complet, prédictions, sauvegarde/chargement joblib). |

### Règles anti-fuite

- Les variables d'un match sont figées avant que son score ou son box-score ne mettent à jour l'état des équipes (test : modifier le résultat d'un match ne change aucune variable de ce match ni des précédents).
- Les matchs à venir ne mettent jamais à jour l'historique.
- Le découpage walk-forward ne coupe jamais une journée entre entraînement et test.
- Seules les cotes `verified_point_in_time` capturées strictement avant `scheduled_tipoff_at` sont utilisées. La cote de marché n'entre comme variable que si au moins 60 % des matchs terminés en disposent.

### Exécution

Depuis `nba_quant_engine_collector/nba_quant_engine` :

    python -m engine.cli evaluate --source db                    # backtest sur Supabase
    python -m engine.cli predict --source db                     # probabilités des matchs à venir
    python -m engine.cli predict --source db --save-model artifacts/engine.joblib
    python -m engine.cli evaluate --source demo                  # ligue synthétique, hors ligne
    python -m engine.cli evaluate --source fivethirtyeight --path nbaallelo.csv --first-season 2004 --min-train-games 2400 --retrain-every 300
    python -m engine.cli evaluate --source csv --games games.csv --team-stats team_stats.csv --odds odds.csv

`evaluate` affiche un résumé sur stderr et écrit le rapport JSON sur stdout (ou dans `--output`). La page **Moteur prédictif** du tableau de bord fait le même calcul et fonctionne aussi en mode démo sans secrets Supabase.

### Résultats de référence (vraies données NBA)

Backtest walk-forward sur l'historique public FiveThirtyEight (scores uniquement, sans box-scores ni cotes), saisons 2004 à 2015, premier entraînement sur 2 400 matchs, réentraînement tous les 300 matchs : **13 084 matchs prédits hors échantillon**.

| Modèle | Précision | Brier | Log-loss | AUC |
| --- | --- | --- | --- | --- |
| Ensemble (moteur) | 67,5 % | 0,2059 | 0,598 | 0,721 |
| Régression logistique | 67,6 % | 0,2059 | 0,598 | 0,721 |
| Marge → probabilité | 67,5 % | 0,2062 | 0,598 | 0,721 |
| Gradient boosting | 67,2 % | 0,2085 | 0,604 | 0,713 |
| Elo seul | 67,2 % | 0,2072 | 0,601 | 0,719 |
| Toujours domicile | 60,0 % | 0,2401 | 0,673 | 0,500 |

Erreur absolue moyenne sur la marge : 9,4 points. La calibration est bonne (par exemple, les matchs annoncés entre 80 et 90 % sont gagnés à 84 %). Sur ce jeu de données, les variables issues des box-scores sont absentes ; avec les box-scores DunkAPI, les four factors et ratings par possession sont actifs. Pour comparaison, les marchés de paris NBA atteignent généralement une précision de 68 à 70 % : battre durablement les cotes de clôture reste très difficile.

### Résultats sur la base Supabase (novembre 2026)

Backtest walk-forward sur les saisons 2023-24 à 2026-27 collectées (box-scores DunkAPI) : 3 408 matchs prédits hors échantillon.

| Modèle | Précision | Brier | AUC |
| --- | --- | --- | --- |
| Marge → probabilité | 67,0 % | 0,2086 | 0,729 |
| Ensemble (avec gradient boosting, v1.0) | 67,0 % | 0,2092 | 0,728 |
| Régression logistique | 66,5 % | 0,2092 | 0,727 |
| Elo seul | 65,8 % | 0,2144 | 0,728 |
| Gradient boosting | 65,3 % | 0,2170 | 0,710 |
| Toujours domicile | 54,4 % | 0,2481 | 0,500 |

Le moteur bat l'Elo à chaque saison (2024 : 65,5 % contre 64,4 % ; 2025 : 67,1 % contre 65,2 % ; 2026 : 67,8 % contre 67,6 %).

### Page « Prochains matchs »

- Noms complets des équipes et heure de Paris.
- Colonne **Accord modèles** : 🟢 Fort (tous les modèles et la marge désignent le même vainqueur, à moins de 10 points d'écart), 🟡 Moyen, 🔴 Faible (au moins un modèle désigne l'autre équipe : à éviter).
- **Calculateur de value bet** : saisissez les cotes décimales de votre bookmaker ; la page affiche l'EV de chaque côté, le conseil et la mise en euros (bankroll, fraction de Kelly, mise maximale et EV minimum réglables).

### Limites

- Pas encore de données joueurs (blessures, absences, minutes), qui expliquent une grande partie de l'écart restant avec le marché.
- Le backtest de mises n'a de sens qu'avec un historique de cotes PIT suffisant : sans cotes, il est simplement ignoré.
- Une performance hors échantillon passée n'est pas une garantie de rentabilité future.
