# REGISTRE DES DÉCISIONS D'ARCHITECTURE & TECHNIQUES (ADR)

Ce document consigne chaque arbitrage technique structurant pris tout au long du projet Smart Predict, avec son contexte, sa justification et ses conséquences.

---

## ADR-001 : Choix du moteur de base de données relationnelle (SQLite)

- **Date** : Octobre 2026
- **Statut** : Validé
- **Décideurs** : Équipe Projet (Ingénieur & Assistant)

### Contexte
Le guide technique du projet prévoit le stockage des données nettoyées dans une base relationnelle (SQLite ou MySQL) afin de simuler un environnement de production où les flux de données proviennent d'une source SQL transactionnelle / analytique.

### Décision
Nous choisissons **SQLite** comme système de gestion de base de données (SGBD).

### Justification & Rationale Senior
1. **Zéro dépendance d'infrastructure** : SQLite est une base de données autonome stockée sous forme de fichier local. Elle ne requiert aucun serveur tiers en tâche de fond (contrairement à MySQL nécessitant XAMPP, un service Windows ou un conteneur Docker).
2. **Reproductibilité & Portabilité immédiate** : N'importe quel développeur, évaluateur ou recruteur clonant le dépôt GitHub peut exécuter l'ensemble du pipeline sans aucune étape préalable d'installation ou de configuration réseau/utilisateur.
3. **Compatibilité native Python** : SQLite est directement intégré à la bibliothèque standard Python (`sqlite3`), éliminant les risques de dérive de drivers ou de connecteurs natifs.
4. **Adéquation au volume** : Avec ~10 886 lignes de données horaires, SQLite offre des performances d'indexation et d'agrégation instantanées, largement suffisantes pour le cas d'usage.

### Conséquences
- Fichier de base de données localisé dans `data/database.db` (ou `data/processed/database.db`).
- Ce fichier de base est exclu du suivi Git via `.gitignore` pour éviter le stockage de binaires volumineux modifiés fréquemment, tandis que le script de construction SQL (`sql/schema.sql` et `src/ingestion_sql.py`) garantit sa reproductibilité déterministe.

---

## ADR-002 : Transformation logarithmique de la variable cible count

- **Date** : Octobre 2026
- **Statut** : Validé (vérifié dans `notebooks/01_eda.ipynb`)
- **Décideurs** : Équipe Projet

### Contexte & Constat chiffré
L'analyse exploratoire de la distribution de `count` montre une forte asymétrie positive (skewness de **+1.2421**), avec une moyenne à 191.57, une médiane à 145.0 et un pic à 977 locations/h.

Une comparaison des transformations de stabilisation donne :
- **Skewness brut** : **+1.2421** (forte asymétrie à droite)
- **Skewness racine carrée $\sqrt{\text{count}}$** : **+0.2576** (distribution quasi-symétrique, nettement plus proche de 0)
- **Skewness logarithmique $\log(1 + \text{count})$** : **-0.8514** (sur-correction induisant une asymétrie à gauche)

### Décision
1. Retenir $\log(1 + \text{count})$ et $\sqrt{\text{count}}$ comme deux variantes candidates pour la modélisation. Le choix définitif ne se fera pas sur le seul critère de skewness, mais sera formellement départagé en J3/J4 sur la base des erreurs réelles (RMSE / MAE / RMSLE).
2. **Distinction entre perte d'entraînement et métriques de restitution** :
   - L'erreur minimisée pendant l'entraînement sur données transformées ne doit pas être confondue avec la performance métier finale.
   - Les prédictions seront obligatoirement ré-inversées sur l'échelle réelle (ex: `expm1(y_pred)` ou `(y_pred)**2` avec écrêtage à $\ge 0$).
   - **Restitution à double niveau** :
     - **RMSE et MAE sur l'échelle réelle** (exprimés en nombre concret de vélos) pour l'interprétabilité métier, le dimensionnement de la flotte et le dashboard Power BI.
     - **RMSLE** calculé et reporté séparément comme métrique d'évaluation relative et de comparaison standard.

### Justification & Rationale Senior
1. **Stabilisation de la variance** : $\sqrt{y}$ symétrise fortement la distribution (+0.2576) sans sur-corriger vers la gauche comme $\log(1+y)$ (-0.8514). Cependant, l'évaluation de leur supériorité prédictive doit reposer sur l'erreur constatée après ré-inversion réelle.
2. **Alignement métrique** : Optimiser l'erreur quadratique sur $\log(1 + y)$ oriente l'apprentissage vers la réduction des erreurs relatives (RMSLE), évitant que le modèle ne sur-apprenne uniquement sur les heures de très forte affluence au détriment des heures creuses.
3. **Interprétabilité opérationnelle** : Un décideur ou un régulateur de flotte sur Power BI a besoin d'évaluer les écarts en vélos réels (RMSE/MAE échelle brute), tandis que l'ingénieur ML suit le RMSLE pour vérifier l'adéquation mathématique globale.
4. **Garantie de positivité** : La ré-inversion (avec écrêtage à 0) empêche tout risque de prédiction de volumes négatifs.

---

## ADR-003 : Exclusion stricte des variables casual et registered (Anti-Data Leakage)

- **Date** : Octobre 2026
- **Statut** : Validé (vérifié dans `notebooks/01_eda.ipynb`)
- **Décideurs** : Équipe Projet

### Contexte & Constat chiffré
Le test logique `(df['casual'] + df['registered'] == df['count']).all()` renvoie **`True`** sur les **10 886 lignes** (0 discordance).

### Décision
Exclusion totale et stricte de `casual` et `registered` du vecteur de variables explicatives ($X$).

### Justification & Rationale Senior
Ces deux colonnes constituent la partition exacte de la cible. En conditions réelles de production, ni le nombre d'abonnés ni celui de non-abonnés ne sont connus avant l'heure d'inférence. Les conserver constituerait une **fuite de données (Data Leakage) absolue** rendant le modèle inopérant en production.

---

## ADR-004 : Arbitrage de colinéarité entre température réelle et ressentie

- **Date** : Octobre 2026
- **Statut** : Validé (vérifié dans `notebooks/01_eda.ipynb`)
- **Décideurs** : Équipe Projet

### Contexte & Constat chiffré
La corrélation de Pearson entre `temp` et `atemp` est mesurée à **0.9849**.

### Décision
Ne pas inclure conjointement les deux variables brutes dans le modèle final. Retenir `temp` en priorité (ou tester la différence `atemp - temp` comme indicateur de ressenti thermique).

### Justification & Rationale Senior
Une colinéarité aussi extrême (0.985) dégrade l'interprétabilité des coefficients dans les modèles linéaires et dilue inutilement l'importance des variables dans les modèles ensemblistes sans apporter d'information nouvelle.

---

## ADR-005 : Prise en compte de l'anomalie de capteur sur la vitesse du vent (windspeed) & Corrélation humidité

- **Date** : Octobre 2026
- **Statut** : Validé (vérifié dans `notebooks/01_eda.ipynb`)
- **Décideurs** : Équipe Projet

### Contexte & Constat chiffré
- Exactement **1 313 lignes (12.06%)** présentent la valeur `windspeed == 0.0`.
- La corrélation de Pearson entre `humidity` et `count` est mesurée à **-0.3174** (relation négative modérée).

### Décision
1. **windspeed** : Documenter ces 1 313 zéros comme une anomalie probable de mesure (hypothèse de valeurs non mesurées ou sous le seuil de sensibilité d'anémomètre codées à 0.0). La décision de traitement (imputation, modèle de régression du vent, ou maintien) est **différée au feature engineering (J4) et non tranchée ici**.
2. **humidity** : Conserver la variable `humidity` comme feature prédictive continue continue (impact négatif modéré sur la demande).

---

## ADR-006 : Modélisation relationnelle SQL, contraintes d'intégrité CHECK et idempotence

- **Date** : Octobre 2026
- **Statut** : Validé (validé par `tests/test_sql_integrity.py`)
- **Décideurs** : Équipe Projet

### Contexte & Constat
La simulation d'un flux de données de production impose de stocker les données historiques dans une base relationnelle locale (`data/database.db`) servant de source de vérité pour le futur pipeline ML.

### Décision
1. **Schéma DDL strict (`sql/schema.sql`)** :
   - Stockage de `datetime` en `TEXT NOT NULL UNIQUE` au format normalisé ISO-8601 (`YYYY-MM-DD HH:MM:SS`).
   - Application de contraintes `CHECK` physiques et logiques : `season BETWEEN 1 AND 4`, `weather BETWEEN 1 AND 4`, `workingday IN (0, 1)`, `holiday IN (0, 1)`, bornes thermiques plausibles, `humidity BETWEEN 0.0 AND 100.0`.
   - Contrainte inter-colonnes `CONSTRAINT chk_total_consistency CHECK (casual + registered = count)` garantissant la cohérence mathématique des comptages à l'ingestion.
   - Création d'un index B-tree `idx_bike_trips_datetime` sur `datetime` pour optimiser les requêtes chronologiques du pipeline ML.
2. **Ingestion transactionnelle et idempotente (`src/ingestion_sql.py`)** :
   - Recréation idempotente de la table via `DROP TABLE IF EXISTS`.
   - Insertion par curseur paramétré `executemany` (exclusion de `pandas.to_sql` pour respecter rigoureusement les contraintes `CHECK`).
   - Encapsulation dans une transaction unique avec vérification de l'assertion volumétrique (exactement 10 886 lignes) **dans la transaction avant le COMMIT** (avec `ROLLBACK` automatique si anomalie).
3. **Statistiques météo 4** :
   - Documenté que la classe `weather = 4` (Pluie forte/Orage) ne comporte qu'**une unique observation (N=1, count=164)**. Ce groupe est statistiquement négligeable et ne doit pas faire l'objet de conclusions ou d'assertions de représentativité sur sa moyenne.

### Justification & Rationale Senior
Ce niveau de contraintes transforme la base SQL en pare-feu contre la corruption de données (Data Quality at Ingestion), empêchant l'insertion silencieuse de lignes incohérentes ou de doublons d'horodatage.

---

## ADR-007 : Découpage temporel chronologique, trou calendaire et benchmark des baselines

- **Date** : Octobre 2026
- **Statut** : Validé (validé par `tests/test_baseline.py`)
- **Décideurs** : Équipe Projet

### Contexte & Spécificité du dataset
Le dataset Kaggle historique s'étend du 01/01/2011 au 19/12/2012, avec une collecte arrêtée au 19 de chaque mois (les jours 20 à la fin du mois constituant le test set privé Kaggle).

### Décision & Chiffres mesurés
1. **Découpage chronologique strict** :
   - Règle de filtrage : `Train = df[datetime < '2012-07-01']` et `Test = df[datetime >= '2012-07-01']`.
   - **Volumétrie exacte** :
     - **Train** : **8 151 lignes** (**74.88%** du total).
     - **Test** : **2 735 lignes** (**25.12%** du total).
   - **Frontières et trou calendaire** :
     - Date maximale d'entraînement : **`2012-06-19 23:00:00`**.
     - Date minimale de test : **`2012-07-01 00:00:00`**.
     - **Trou calendaire explicite** : Du **20/06/2012 au 30/06/2012 inclus** (11 jours sans données), inhérent à la structure Kaggle. Cet intervalle de 11 jours réduit fortement la dépendance d'autocorrélation à court terme entre la fin du train et le début du test, sans toutefois éliminer l'autocorrélation globale à plus long terme (saisonnalité, tendance annuelle).
2. **Benchmark des Baselines (évaluées sur le jeu de test de 2 735 lignes)** :
   - **Baseline 0 (Moyenne globale Train = 167.7858)** :
     - RMSE : **235.9797** vélos
     - MAE  : **178.2798** vélos
     - RMSLE: **1.3598**
   - **Baseline 1 (Heuristique `hour x workingday`)** :
     - RMSE : **144.4269** vélos
     - MAE  : **102.1689** vélos
     - RMSLE: **0.5304**
   - **Gain mesuré apporté par l'information temporelle** :
     - Réduction de l'erreur absolue (RMSE) : **-91.5528 vélos (-38.80%)**.
     - Réduction de l'erreur logarithmique (RMSLE) : **-0.8294 (-60.99%)**.

### Justification & Rationale Senior
1. Comparer systématiquement la baseline heuristique à la moyenne naïve prouve mathématiquement l'apport de la variable horaire et du calendrier avant d'introduire des algorithmes sophistiqués.
2. Les scores de Baseline 1 constituent le **benchmark contractuel minimal** : tout modèle de Machine Learning développé en J4 (Random Forest, LightGBM) devra impérativement faire mieux que **RMSLE 0.5304** et **RMSE 144.43** sur ce même jeu de test.

---

## ADR-008 : Feature Engineering anti-fuite, modélisation avancée et sélection du modèle champion

- **Date** : Octobre 2026
- **Statut** : Validé (sélection stricte sur validation, évaluation finale sur test neutre)
- **Décideurs** : Équipe Projet

### Contexte & Méthodologie anti-biais
Pour garantir une modélisation sans fuite méthodologique, le jeu de test chronologique (2 735 lignes) est maintenu strictement **gelé comme juge neutre**. Aucun choix d'architecture ou de transformation cible n'est guidé par les scores de test.

### Protocole de découpage temporel à 3 niveaux
1. **Train_fit** (01/01/2011 au 19/03/2012) : 6 785 lignes (83.2% du train).
2. **Validation interne** (01/04/2012 au 19/06/2012) : 1 366 lignes (16.8% du train), servant de juge pour le choix du modèle.
3. **Test gelé (juge neutre)** (01/07/2012 au 19/12/2012) : 2 735 lignes, évalué une seule fois à la fin.

### Shootout systématique sur le jeu de validation (4 configurations)
| Configuration candidate | RMSE (vélos) | RMSLE | R² réel |
| :--- | :---: | :---: | :---: |
| `RandomForest_log1p` | 101.12 | 0.4211 | 0.7891 |
| `RandomForest_sqrt` | 94.68 | 0.4043 | 0.8151 |
| `HistGradientBoosting_log1p` (**Champion retenu**) | 74.93 | **0.3354** | 0.8842 |
| `HistGradientBoosting_sqrt` | **67.98** | 0.3395 | **0.9047** |

*Arbitrage* : `HistGradientBoosting_log1p` est sélectionné comme champion pour sa minimisation prioritaire du RMSLE (0.3354), tout en conservant une excellente régression sur l'échelle réelle (RMSE 74.93).

### Ré-entraînement et Évaluation UNIQUE sur Test Neutre (2 735 lignes)
Le modèle champion a été ré-entraîné sur l'intégralité du Train (8 151 lignes) puis évalué une seule fois sur le jeu de test gelé :
- **RMSE test** : **69.4348** vélos (vs contrat Baseline 1 = 144.4269, soit **-51.92%** d'erreur).
- **MAE test** : **44.6095** vélos (vs contrat Baseline 1 = 102.1689, soit **-56.34%** d'erreur).
- **RMSLE test** : **0.3304** (vs contrat Baseline 1 = 0.5304, soit **-37.71%** d'erreur relative).
- **$R^2$ réel test** : **0.8968** (~90% de la variance expliquée sur le futur).

### Persistance & Inférence
- Estimateur sérialisé dans `models/best_model.joblib`.
- Garantie physique stricte : ré-inversion par `expm1` et écrêtage automatique à $\ge 0$ vélos.

---

## ADR-009 : Pipeline d'automatisation de production et dénormalisation pour Power BI

- **Date** : Octobre 2026
- **Statut** : Validé (validé par `tests/test_pipeline.py`)
- **Décideurs** : Équipe Projet

### Contexte
Le passage à la restitution décisionnelle sous Power BI nécessite une table dénormalisée, optimisée pour les analyses en temps réel, évitant aux décideurs de concevoir des mesures DAX complexes pour reconstituer des résidus ou des libellés métier.

### Décision
1. **Orchestrateur autonome (`src/pipeline.py`)** :
   - Script CLI modulaire exécutant la séquence d'inférence en production : connexion SQLite en lecture seule (`mode=ro`), feature engineering anti-fuite, chargement de `best_model.joblib`, inférence déterministe avec ré-inversion et contrainte physique ($\hat{y} \ge 0$).
   - **Principe d'architecture** : Le pipeline **n'entraîne jamais aucun modèle**. Il est un consommateur passif de l'artefact sérialisé `best_model.joblib` garantissant une inférence reproductible et stable en production.
2. **Structure de la table de restitution (`data/processed/powerbi_export.csv`)** :
   - **Volumétrie** : 10 886 lignes $\times$ 26 colonnes, 0 valeur manquante.
   - **Dénormalisation métier** : Libellés explicites pour les slicers Power BI (`season_label`: Printemps, Été, Automne, Hiver ; `weather_label`: Beau, Brouillard, etc. ; `day_name`: Monday à Sunday).
   - **Encodage de restitution (`utf-8-sig`)** : Utilisation du standard UTF-8 avec BOM (`utf-8-sig`) pour assurer une restitution parfaite des caractères accentués français dans Power BI Desktop et Excel sous Windows.
   - **Rôle des colonnes `casual` et `registered`** : Conservées dans l'export final **exclusivement à des fins d'analyse descriptive et métier sous Power BI** (comprendre la typologie d'usagers abonnés vs occasionnels). Elles ne participent à aucun moment au calcul de `count_predicted` (zéro fuite) et ne doivent pas être confondues avec des variables de prédiction.
   - **Pré-calcul des métriques d'erreur** :
     - `residual = count_real - count_predicted` (signé : permet de détecter la sous-estimation ou sur-estimation par heure/météo).
     - `absolute_error = abs(residual)` (facilite la mesure du MAE moyen).
     - `squared_error = residual ** 2` (facilite la mesure du RMSE : $\sqrt{\text{moyenne}}$).
   - **Partitionnement temporel (`dataset_split`)** : Colonne contenant `'Train'` (8 151 lignes) ou `'Test'` (2 735 lignes), permettant de filtrer instantanément dans Power BI les performances sur le jeu de test neutre ou sur l'historique complet.
   - **Règle méthodologique de reporting** : Les lignes étiquetées `Train` servent uniquement à la visualisation de la continuité historique ; **elles ne doivent en aucun cas être utilisées pour mesurer ou communiquer la performance du modèle** (seul le sous-ensemble `Test` fait foi pour évaluer l'erreur réelle en conditions réelles de production : RMSE 69.43 vélos, RMSLE 0.3304).

### Justification & Rationale Senior
Pré-calculer les résidus à la source du pipeline garantit l'alignement mathématique strict entre l'environnement Python et le reporting Power BI, tout en maximisant la fluidité des interactions et des slicers pour l'utilisateur final.


