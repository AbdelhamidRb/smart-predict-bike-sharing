# Smart Predict : Pipeline de Prédiction de Demande de Vélos (Bike Sharing Demand)

[![CI - Test Suites](https://img.shields.io/badge/pytest-passing-success.svg)](tests/)
[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](requirements.txt)
[![SQLite](https://img.shields.io/badge/Database-SQLite3-lightgrey.svg)](sql/)
[![Power BI](https://img.shields.io/badge/Reporting-Power%20BI-yellow.svg)](powerbi/)

Projet conçu selon les **standards d'ingénierie logicielle et de data science de niveau senior** : modélisation relationnelle transactionnelle, absence formelle de fuite de données (*Data Leakage*), sélection de modèles sur validation interne, protocole de test automatisé strict et restitution interactive sous Power BI Desktop.

---

## 1. Contexte & Problématique Métier

Dans les systèmes de vélos en libre-service (*Bike Sharing*), l'enjeu opérationnel majeur réside dans la **régulation de la flotte** : éviter la saturation des stations en heures de pointe tout en garantissant la disponibilité des vélos en heures creuses.

- **Objectif** : Prédire le volume total de locations horaires (`count`) à partir de variables calendaires et météorologiques.
- **Source** : Données historiques Kaggle *Bike Sharing Demand* (10 886 observations horaires).

---

## 2. Architecture Globale du Pipeline

Le pipeline industriel s'articule en 4 couches découplées et reproductibles :

```
[ Données Brutes ] (data/raw/train.csv)
        │
        ▼
[ Ingestion & Intégrité SQL ] (src/ingestion_sql.py -> data/database.db)
        │
        ▼
[ Feature Engineering Anti-Fuite ] (src/features.py)
        │
        ▼
[ Entraînement & Sélection ML ] (src/train.py -> models/best_model.joblib)
        │  • Découpage à 3 niveaux : Train_fit / Validation interne / Test gelé
        ▼
[ Pipeline Autonome d'Inférence ] (src/pipeline.py)
        │
        ▼
[ Restitution Décisionnelle ] (data/processed/powerbi_export.csv -> Power BI Desktop)
```

---

## 3. Rigueur d'Ingénierie & Règles Anti-Fuite

| Défi Méthodologique | Risque Critique | Solution Senior Implémentée |
| :--- | :--- | :--- |
| **Fuite temporelle (Time Leakage)** | `train_test_split` avec mélange aléatoire (`shuffle=True`) qui entraîne sur le futur pour prédire le passé. | **Découpage chronologique strict** :<br>• `Train_fit` (01/2011 au 03/2012 : 6 785 obs)<br>• `Validation` (04/2012 au 06/2012 : 1 366 obs)<br>• `Test gelé neutre` (07/2012 au 12/2012 : 2 735 obs). |
| **Fuite de cible (Target Leakage)** | Les colonnes `casual` (non-abonnés) et `registered` (abonnés) somment exactement à `count` (`casual + registered == count` vérifié à 100%). | **Exclusion formelle** de `casual` et `registered` de toutes les matrices d'entraînement ML (**ADR-003**). |
| **Colinéarité extrême** | Corrélation $r = 0.9849$ entre `temp` et `atemp` (température ressentie). | Exclusion d'`atemp` pour éliminer la redondance et stabiliser l'apprentissage (**ADR-004**). |
| **Qualité des données à l'ingestion** | Insertion de données corrompues ou de doublons d'horodatage. | Schéma relationnel strict (`sql/schema.sql`) avec contraintes `CHECK`, indexation B-tree sur `datetime`, et vérification volumétrique dans une transaction atomique (**ADR-006**). |
| **Garantie physique** | Prédire des volumes de vélos négatifs. | Modélisation sur variable transformée $\log(1 + y)$ et ré-inversion bornée par `np.clip(..., 0, None)`. |

---

## 4. Benchmark des Résultats & Contrats de Performance

Conformément aux bonnes pratiques, **deux baselines heuristiques** ont été établies pour fixer le contrat de performance minimal que les modèles ML devaient impérativement battre sur le jeu de test neutre :

| Modèle / Approche | Description | RMSE (vélos) | MAE (vélos) | RMSLE | Gain vs Baseline 0 |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Baseline 0** | Moyenne globale naïve ($167.79$) | 235.98 | 178.28 | 1.3598 | Benchmark initial |
| **Baseline 1** | Heuristique `hour x workingday` | 144.43 | 102.17 | 0.5304 | -38.8% RMSE |
| **Random Forest (`sqrt`)** | Ensembliste bagging | 85.30 | 55.56 | 0.3664 | -63.8% RMSE |
| **HistGradientBoosting (`log1p`)** | **Modèle Champion Retenu** | **69.43** | **44.61** | **0.3304** | **-70.6% RMSE** |

> **Résultat Champion** : L'algorithme **`HistGradientBoosting`** surpasse la Baseline 1 de **-51.9% d'erreur RMSE** et atteint un coefficient de détermination **$R^2 = 89.68\%$** sur le jeu de test gelé.

---

## 5. Dashboard Décisionnel Power BI

Le fichier Power BI Desktop (`powerbi/smart_predict_dashboard.pbix`) est connecté au flux exporté (`data/processed/powerbi_export.csv`, encodage `utf-8-sig`) :

- **Cartes KPI dynamiques** :
  - `RMSE Test` : **69.44** vélos
  - `MAE Test` : **44.61** vélos
  - `Volume Réel Test` : **717.85 K** vélos
  - `Volume Prédit Test` : **642.96 K** vélos
- **Analyses visuelles** :
  - *Courbe temporelle Réel vs Prédit* : Suivi continu des tendances journalières et saisonnières.
  - *Profil horaire pendulaire* : Mise en évidence des pics domicile-travail (8h et 17h) en jour ouvré vs plateau récréatif le week-end.
  - *Distribution des résidus ($y - \hat{y}$)* : Distribution centrée sur 0 démontrant l'absence de biais systématique.
- **Slicers interactifs** : Filtrage par partition (`Test` vs `Train`), saison, conditions météorologiques et année.

---

## 6. Structure du Répertoire

```text
smart-predict-bike-sharing/
├── data/
│   ├── raw/                 # Fichier source Kaggle (train.csv)
│   └── processed/           # Export dénormalisé pour Power BI (powerbi_export.csv)
├── sql/
│   ├── schema.sql           # Schéma DDL DDL strict avec contraintes CHECK
│   └── analytical_queries.sql # Requêtes analytiques de validation
├── src/
│   ├── ingestion_sql.py     # Script d'ingestion transactionnelle SQLite
│   ├── split.py             # Découpage chronologique étanche sans fuite
│   ├── baseline.py          # Calcul des baselines heuristiques et métriques
│   ├── features.py          # Feature engineering (sin/cos, calendrier, météo)
│   ├── train.py             # Entraînement, shootout sur validation et sérialisation
│   ├── pipeline.py          # Pipeline autonome d'inférence en production
│   └── check_powerbi_export.py # Contrôle qualité de l'export Power BI
├── notebooks/
│   ├── 01_eda.ipynb         # Notebook complet d'analyse exploratoire exécuté
│   └── 01_eda.md            # Export Markdown du notebook
├── models/
│   └── best_model.joblib    # Modèle champion sérialisé
├── powerbi/
│   ├── smart_predict_dashboard.pbix # Fichier Power BI Desktop final
│   └── GUIDE_POWERBI.md     # Guide pas-à-pas et formules DAX
├── tests/
│   ├── test_sql_integrity.py # 6 tests d'intégrité de la base relationnelle
│   ├── test_baseline.py      # Tests du calcul miniature à la main et de la frontière
│   ├── test_modeling.py      # Tests du contrat de performance du modèle champion
│   └── test_pipeline.py      # Tests de bout en bout de l'export de production
├── CONTEXTE.md              # Cadrage métier, architecture et conventions
├── DECISIONS.md             # Registre des 10 Décisions d'Architecture (ADR-001 à ADR-010)
├── requirements.txt         # Dépendances Python verrouillées
└── README.md                # Documentation générale du projet
```

---

## 7. Guide de Reproduction Rapide

### Prérequis
- Python 3.10+
- Git

### Installation
```bash
# 1. Cloner le dépôt
git clone https://github.com/AbdelhamidRb/smart-predict-bike-sharing.git
cd smart-predict-bike-sharing

# 2. Créer et activer l'environnement virtuel
python -m venv venv
# Windows :
.\venv\Scripts\activate

# 3. Installer les dépendances
pip install -r requirements.txt
```

### Exécution du Pipeline de Bout en Bout
```bash
# Ingestion dans SQLite
python src/ingestion_sql.py

# Calcul des baselines
python src/baseline.py

# Entraînement et sélection du modèle champion
python src/train.py

# Inférence de production et export Power BI
python src/pipeline.py
```

### Lancement des Suites de Tests Automatisés
```bash
pytest tests/ -v
```
*(17 tests unitaires et d'intégration validant la conformité SQL, mathématique, ML et pipeline).*
