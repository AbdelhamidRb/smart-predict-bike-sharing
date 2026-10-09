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

