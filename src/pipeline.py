"""Pipeline autonome d'ingénierie et d'inférence en production.

Orchestre la chaîne complète :
1. Extraction depuis SQLite (mode=ro).
2. Feature engineering sans fuite temporelle.
3. Chargement de l'estimateur champion (models/best_model.joblib).
4. Inférence sur l'ensemble de la plage temporelle avec ré-inversion et contrainte physique (>= 0).
5. Dénormalisation des libellés métier (saison, météo) et calcul amont des résidus pour Power BI.
6. Export déterministe dans data/processed/powerbi_export.csv.
"""

from pathlib import Path
import argparse
import logging
import time
import joblib
import numpy as np
import pandas as pd

from src.split import load_data_from_db, SPLIT_DATE_THRESHOLD
from src.features import extract_features

# Configuration du logging structuré
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("ProductionPipeline")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "database.db"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "best_model.joblib"
DEFAULT_OUTPUT_CSV = PROJECT_ROOT / "data" / "processed" / "powerbi_export.csv"

# Dictionnaires de libellés pour restitution décisionnelle Power BI
SEASON_LABELS = {
    1: "Printemps",
    2: "Été",
    3: "Automne",
    4: "Hiver",
}

WEATHER_LABELS = {
    1: "Beau / Dégagé",
    2: "Brouillard / Nuageux",
    3: "Neige / Pluie légère",
    4: "Pluie forte / Orage",
}


def run_pipeline(
    db_path: Path = DEFAULT_DB_PATH,
    model_path: Path = DEFAULT_MODEL_PATH,
    output_csv: Path = DEFAULT_OUTPUT_CSV,
) -> pd.DataFrame:
    """Exécute le pipeline complet de bout en bout et génère le fichier Power BI."""
    start_time = time.time()
    logger.info("Démarrage du pipeline de production Smart Predict...")

    # 1. Validation de l'existence des dépendances amont
    if not db_path.exists():
        raise FileNotFoundError(f"Base de données source introuvable : {db_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Modèle sérialisé introuvable : {model_path}")

    # 2. Extraction des données
    logger.info("Étape 1/5 : Extraction des données depuis SQLite (%s)...", db_path.name)
    df_raw = load_data_from_db(db_path)
    total_rows = len(df_raw)
    logger.info("Données extraites avec succès : %d observations.", total_rows)

    # 3. Feature engineering
    logger.info("Étape 2/5 : Génération des features anti-fuite...")
    X = extract_features(df_raw)
    logger.info("Matrice de features générée : forme %s.", X.shape)

    # 4. Chargement du modèle champion
    logger.info("Étape 3/5 : Chargement du modèle champion depuis %s...", model_path.name)
    model_payload = joblib.load(model_path)
    estimator = model_payload["estimator"]
    target_transform = model_payload["target_transform"]
    model_name = model_payload.get("model_name", "ChampionModel")
    logger.info("Modèle chargé : %s (transformation cible : %s).", model_name, target_transform)

    # 5. Inférence et ré-inversion stricte
    logger.info("Étape 4/5 : Exécution des prédictions sur les %d observations...", total_rows)
    preds_raw = estimator.predict(X)

    if target_transform == "log1p":
        preds_real = np.expm1(preds_raw)
    elif target_transform == "sqrt":
        preds_real = np.square(np.clip(preds_raw, 0, None))
    else:
        raise ValueError(f"Transformation de cible non supportée : {target_transform}")

    # Garantie physique absolue : aucun volume de vélos négatif
    preds_real = np.clip(preds_real, 0, None)
    # Arrondi à 1 décimale pour concilier précision continue et lisibilité Power BI
    predicted_count = np.round(preds_real, 1)

    # 6. Construction de la table de restitution enrichie pour Power BI
    logger.info("Étape 5/5 : Construction de la table dénormalisée pour Power BI...")
    dt = df_raw["datetime"]
    split_threshold = pd.Timestamp(SPLIT_DATE_THRESHOLD)

    export_df = pd.DataFrame(index=df_raw.index)

    # Axes temporels
    export_df["datetime"] = dt.dt.strftime("%Y-%m-%d %H:%M:%S")
    export_df["year"] = dt.dt.year
    export_df["month"] = dt.dt.month
    export_df["day"] = dt.dt.day
    export_df["hour"] = dt.dt.hour
    export_df["dayofweek"] = dt.dt.dayofweek
    export_df["day_name"] = dt.dt.day_name()

    # Calendrier et statuts
    export_df["workingday"] = df_raw["workingday"]
    export_df["holiday"] = df_raw["holiday"]
    export_df["season"] = df_raw["season"]
    export_df["season_label"] = df_raw["season"].map(SEASON_LABELS)

    # Météo dénormalisée
    export_df["weather"] = df_raw["weather"]
    export_df["weather_label"] = df_raw["weather"].map(WEATHER_LABELS)
    export_df["temp"] = df_raw["temp"]
    export_df["atemp"] = df_raw["atemp"]
    export_df["humidity"] = df_raw["humidity"]
    export_df["windspeed"] = df_raw["windspeed"]

    # Comptages réels
    export_df["casual"] = df_raw["casual"]
    export_df["registered"] = df_raw["registered"]
    export_df["count_real"] = df_raw["count"]

    # Prédictions du modèle
    export_df["count_predicted"] = predicted_count

    # Résidus et métriques ligne à ligne (évite des formules DAX complexes)
    residuals = export_df["count_real"] - export_df["count_predicted"]
    export_df["residual"] = np.round(residuals, 1)
    export_df["absolute_error"] = np.round(np.abs(residuals), 1)
    export_df["squared_error"] = np.round(residuals ** 2, 2)

    # Tag de partition temporelle (Train historique vs Test neutre)
    export_df["dataset_split"] = np.where(dt < split_threshold, "Train", "Test")

    # Métadonnées d'exécution
    export_df["model_version"] = model_name

    # 7. Contrôles d'intégrité de sortie
    assert len(export_df) == total_rows, f"Volumétrie incohérente : {len(export_df)} != {total_rows}"
    assert export_df.isna().sum().sum() == 0, "Présence inattendue de valeurs manquantes dans l'export."
    assert (export_df["count_predicted"] >= 0).all(), "Prédictions négatives détectées dans l'export."

    # 8. Export CSV déterministe
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    export_df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    elapsed = time.time() - start_time

    logger.info(
        "Export terminé avec succès : %d lignes écrites dans %s (durée : %.2fs).",
        len(export_df),
        output_csv,
        elapsed,
    )
    return export_df


def main():
    """Point d'entrée CLI pour orchestration manuelle ou automatisée."""
    parser = argparse.ArgumentParser(description="Pipeline autonome Smart Predict")
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH, help="Chemin vers la base SQLite")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH, help="Chemin vers best_model.joblib")
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV, help="Chemin de sortie du CSV")
    args = parser.parse_args()

    run_pipeline(
        db_path=args.db_path,
        model_path=args.model_path,
        output_csv=args.output_csv,
    )


if __name__ == "__main__":
    main()
