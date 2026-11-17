# NBA Quant Engine

Collecteur NBA DunkAPI → Supabase et tableau de bord Streamlit.

## Démarrer l'application

L'application Streamlit se lance depuis la racine du dépôt :

    streamlit run streamlit_app.py

Pour déployer sur Streamlit Community Cloud :

- Repository : azmourad067-stack/NBA-Quant-Engine
- Branch : feature/streamlit-dashboard
- Main file path : streamlit_app.py
- Python : 3.12

Les dépendances principales sont déclarées dans le requirements.txt de la racine, qui inclut celui du collecteur.

## Secrets à configurer dans Streamlit Cloud

Dans App → Settings → Secrets, ajouter :

    APP_ACCESS_PASSWORD = "un-mot-de-passe-long-et-unique"
    DUNKAPI_API_KEY = "YOUR_DUNKAPI_API_KEY"
    SUPABASE_DB_URL = "postgresql://postgres...."
    NBA_DB_SCHEMA = "nba"
    DUNKAPI_BASE_URL = "https://mcp.dunkapi.com"
    MAX_CREDITS_PER_RUN = "350"

Utiliser l'URI PostgreSQL obtenue dans Supabase → Connect → Session Pooler. Ne jamais utiliser ni publier la clé service_role à la place de cette URI. Le mot de passe APP_ACCESS_PASSWORD est obligatoire : l'application refuse de s'ouvrir tant qu'il n'est pas défini. Ne jamais committer les secrets.

## Fonctions actuelles

- Vue d'ensemble : comptages des tables, derniers matchs, qualité PIT des cotes et historique des imports.
- Collecteur : lancement manuel par saison, statut, taille de lot et plafond de crédits.
- Conservation du collecteur CLI et des fonctions d'import existantes.
- Validation du schéma avant écriture ; upserts idempotents ; audit des appels et conservation des payloads bruts.

L'ouverture du tableau de bord ne déclenche aucune collecte. Commencer par un petit lot et vérifier les données et les crédits consommés avant d'augmenter la charge.

Voir nba_quant_engine_collector/nba_quant_engine/README.md pour les instructions locales détaillées et les règles d'intégrité des données.


## Moteur prédictif v1.0

Le moteur (`nba_quant_engine_collector/nba_quant_engine/engine/`) produit, pour chaque match à venir, la probabilité de victoire, la marge et le total de points attendus, les cotes justes et un éventuel value bet sur cotes PIT. Il combine Elo avec marge de victoire, forme, repos, ratings par possession, four factors, régression logistique, gradient boosting et régression de marge, évalués par un backtest walk-forward sans fuite temporelle.

Sur 13 084 matchs NBA réels prédits hors échantillon (2005–2015, historique public FiveThirtyEight), le moteur atteint 67,5 % de précision et un Brier score de 0,206, contre 0,207 pour l'Elo seul et 0,240 pour « toujours domicile ».

    cd nba_quant_engine_collector/nba_quant_engine
    python -m engine.cli evaluate --source db       # backtest sur Supabase
    python -m engine.cli predict --source db        # prochains matchs
    python -m engine.cli evaluate --source demo     # démo hors ligne

Le détail (architecture, règles anti-fuite, résultats, limites) est dans le README du sous-projet.
