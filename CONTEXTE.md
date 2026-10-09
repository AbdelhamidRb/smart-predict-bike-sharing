# CONTEXTE DU PROJET : SMART PREDICT (BIKE SHARING DEMAND)

## 1. Objectif Métier
Le projet **Smart Predict** vise à concevoir et déployer un pipeline de données de bout en bout pour prédire la demande horaire de vélos en libre-service (`count`). L'enjeu opérationnel est de permettre une allocation optimale de la flotte de vélos (régulation logistique, maintenance prédictive, satisfaction usager).

## 2. Architecture Globale du Pipeline
Le flux de données suit une chaîne industrielle en 4 maillons :
1. **Ingestion & Stockage SQL** : 
   - Récupération du dataset brut Kaggle (`train.csv`).
   - Nettoyage initial et insertion dans une base relationnelle locale (**SQLite**).
   - Validation d'intégrité via requêtes SQL analytiques.
2. **Machine Learning en Python** :
   - Extraction des données depuis la base SQL.
   - Séparation chronologique stricte (pas de fuite temporelle / no data leakage).
   - Établissement d'une baseline naïve (moyenne par heure / type de jour).
   - Entraînement et optimisation de modèles supervisés (Random Forest, Gradient Boosting).
   - Évaluation rigoureuse sur métriques alignées (RMSLE, RMSE, MAE, R²).
3. **Automatisation & Inférence (`pipeline.py`)** :
   - Script modulaire autonome orchestrant l'extraction, l'ingestion, le scoring et l'exportation des résultats comparatifs (valeurs réelles vs prédictions).
4. **Restitution Décisionnelle (Power BI)** :
   - Modélisation des données dans Power BI Desktop.
   - Dashboard interactif présentant les KPIs globaux, le suivi temporel réel vs prédictions, et l'analyse de robustesse via slicers (météo, saison).

## 3. Règles d'Ingénierie & Méthode de Travail
- **Une étape à la fois** : Aucun passage à l'étape suivante sans validation explicite du jalon en cours.
- **Git Flow par Jalon** :
  - Chaque jalon significatif fait l'objet d'une branche dédiée (ex: `j1-eda`, `j2-sql`, etc.).
  - Validation des résultats avant merge dans la branche `main`.
- **Zéro Data Leakage** : Interdiction formelle du `train_test_split` avec mélange aléatoire (`shuffle=True`) sur des données temporelles. Interdiction d'utiliser les cibles dérivées `casual` et `registered` comme variables explicatives pour prédire `count`.
- **Traçabilité & Documentation** :
  - Tout choix d'architecture ou de modélisation doit être consigné dans `DECISIONS.md`.
  - Affichage systématique du diff avant modification de fichier.
