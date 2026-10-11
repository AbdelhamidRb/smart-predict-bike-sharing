# GUIDE DE CONCEPTION DU DASHBOARD POWER BI : SMART PREDICT

Ce document fournit la procédure pas-à-pas pour assembler le dashboard décisionnel dans **Power BI Desktop**, avec les configurations de chargement, les formules DAX de référence et les valeurs attendues exactes pour chaque visuel.

---

## 1. Chargement et Modélisation de la Source de Données

1. Ouvrir **Power BI Desktop**.
2. Cliquer sur **Obtenir des données** $\rightarrow$ **Texte/CSV**.
3. Sélectionner le fichier : `data/processed/powerbi_export.csv`.
4. **Paramètres de détection (Obligatoire)** :
   - **Origine du fichier** : `65001 : Unicode (UTF-8)`.
   - **Délimiteur** : `Virgule`.
   - **Détection du type de données** : `Basé sur les 200 premières lignes`.
5. Cliquer sur **Charger** (ou *Transformer les données* pour vérifier les types) :
   - `datetime` : Date/Heure
   - `year`, `month`, `day`, `hour` : Nombre entier
   - `count_real`, `count_predicted`, `residual`, `absolute_error`, `squared_error` : Nombre décimal / entier.

**Chiffres de contrôle attendus immédiatement après chargement** :
- **Nombre total de lignes** : `10 886`
- **Nombre de colonnes** : `26`

---

## 2. Création des Mesures DAX de Référence

Créer une table dédiée aux mesures (ou ajouter directement les mesures dans la table `powerbi_export`) :

### Mesure 1 : RMSE Test (Erreur Quadratique Moyenne sur Test Set Neutre)
```dax
RMSE Test = 
CALCULATE(
    SQRT(AVERAGE(powerbi_export[squared_error])),
    powerbi_export[dataset_split] = "Test"
)
```
- **Valeur exacte attendue** : **`69.44`** vélos (conforme au test neutre J4 à 69.4351).

---

### Mesure 2 : MAE Test (Erreur Absolue Moyenne sur Test Set Neutre)
```dax
MAE Test = 
CALCULATE(
    AVERAGE(powerbi_export[absolute_error]),
    powerbi_export[dataset_split] = "Test"
)
```
- **Valeur exacte attendue** : **`44.61`** vélos (44.6095).

---

### Mesure 3 : Volume Total Réel (Test)
```dax
Total Real Test = 
CALCULATE(
    SUM(powerbi_export[count_real]),
    powerbi_export[dataset_split] = "Test"
)
```
- **Valeur exacte attendue** : **`717 854`** locations.

---

### Mesure 4 : Volume Total Prédit (Test)
```dax
Total Predicted Test = 
CALCULATE(
    SUM(powerbi_export[count_predicted]),
    powerbi_export[dataset_split] = "Test"
)
```
- **Valeur exacte attendue** : **`642 958.9`** locations.

---

## 3. Configuration des Slicers (Filtres Interactifs)

Placer en haut ou sur le bandeau latéral gauche 4 segments (Slicers) :

1. **Segment `dataset_split`** :
   - Choix : `Train` (8 151 lignes) ou `Test` (2 735 lignes).
   - Recommandation : Sélectionner `Test` par défaut pour analyser les performances de généralisation du modèle.
2. **Segment `season_label`** :
   - Choix : `Automne`, `Hiver` (sur le test set), `Printemps`, `Été` (sur l'historique complet).
3. **Segment `weather_label`** :
   - Choix : `Beau / Dégagé`, `Brouillard / Nuageux`, `Neige / Pluie légère`, `Pluie forte / Orage`.
4. **Segment `year`** :
   - Choix : `2011`, `2012`.

---

## 4. Conception des Visuels & Valeurs de Validation

### Visuel 1 : Cartes KPI (Haut de page)
- **Carte 1** : `[RMSE Test]` $\rightarrow$ Affiche **`69.44`**
- **Carte 2** : `[MAE Test]` $\rightarrow$ Affiche **`44.61`**
- **Carte 3** : `[Total Real Test]` $\rightarrow$ Affiche **`717.85 K`**
- **Carte 4** : `[Total Predicted Test]` $\rightarrow$ Affiche **`642.96 K`**

---

### Visuel 2 : Courbe temporelle Réel vs Prédit
- **Type** : Graphique en courbes (Line Chart).
- **Axe X** : `powerbi_export[datetime]`.
- **Valeurs (Axe Y)** :
  - `count_real` (Somme ou Moyenne)
  - `count_predicted` (Somme ou Moyenne)
- **Comportement attendu** :
  - Fidélité remarquable de la courbe prédite suivant les montées en charge quotidiennes et les pics de fréquentation.
  - En filtrant sur une semaine de juillet ou d'octobre 2012, les deux courbes se superposent étroitement avec un résidu moyen modéré.

---

### Visuel 3 : Profil horaire moyen par type de jour (Ouvré vs Non-ouvré)
- **Type** : Graphique en courbes ou combiné.
- **Axe X** : `powerbi_export[hour]` (0 à 23).
- **Légende** : `powerbi_export[workingday]` (0: Non-ouvré, 1: Ouvré).
- **Valeurs** :
  - `Moyenne de count_real`
  - `Moyenne de count_predicted`
- **Valeurs de contrôle attendues sur le Test set** :
  - À 0h en jour non-ouvré : Réel moyen = **135.4**, Prédit moyen = **105.9**
  - À 0h en jour ouvré     : Réel moyen = **52.7**, Prédit moyen = **40.7**
  - À 1h en jour ouvré     : Réel moyen = **21.9**, Prédit moyen = **16.8**

---

### Visuel 4 : Histogramme de Distribution des Résidus
- **Type** : Graphique à barres ou histogramme groupé.
- **Champ** : `powerbi_export[residual]` groupé par tranches de 25 ou 50 vélos.
- **Valeurs descriptives attendues sur le Test set** :
  - Médiane des résidus : **`+14.10`** vélos.
  - Premier quartile (Q1) : **`-1.10`** vélos.
  - Troisième quartile (Q3) : **`+52.95`** vélos.
  - Écart-type des résidus : **`63.82`** vélos.
  - Forme : Distribution centrée autour de zéro, illustrant l'absence de biais systématique majeur et la concentration des résidus dans l'intervalle $[-50, +50]$.

---

## 5. Sauvegarde de l'Artefact
- Enregistrer le fichier final dans :
  `powerbi/smart_predict_dashboard.pbix`
