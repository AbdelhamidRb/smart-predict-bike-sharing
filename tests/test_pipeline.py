"""Tests d'intégration et de validation pour le pipeline autonome de production (J5).

Vérifie :
1. Génération sans erreur du fichier data/processed/powerbi_export.csv.
2. Volumétrie stricte (10 886 lignes) et complétude (0 valeur nulle).
3. Cohérence mathématique ligne à ligne des résidus :
   - residual = count_real - count_predicted
   - absolute_error = abs(residual)
   - squared_error = residual ** 2
4. Ségrégation exacte des partitions temporelles (8 151 Train / 2 735 Test).
5. Positivité des prédictions (count_predicted >= 0.0).
"""

from pathlib import Path
import math
import numpy as np
import pandas as pd
import pytest

from src.pipeline import run_pipeline, DEFAULT_OUTPUT_CSV

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = DEFAULT_OUTPUT_CSV


def test_1_pipeline_execution_and_file_existence():
    """Test 1 : Exécute le pipeline et vérifie l'existence physique du CSV produit."""
    df_export = run_pipeline()
    assert CSV_PATH.exists(), f"Fichier {CSV_PATH} non généré."
    assert not df_export.empty, "L'export retourné est vide."


def test_2_export_volume_and_schema_completeness():
    """Test 2 : Contrôle volumétrique, typage et absence totale de NaN."""
    df = pd.read_csv(CSV_PATH)

    assert len(df) == 10886, f"Volumétrie incorrecte : {len(df)} != 10886"
    assert df.isna().sum().sum() == 0, f"Présence de valeurs manquantes : {df.isna().sum().to_dict()}"

    expected_columns = [
        "datetime", "year", "month", "day", "hour", "dayofweek", "day_name",
        "workingday", "holiday", "season", "season_label", "weather", "weather_label",
        "temp", "atemp", "humidity", "windspeed",
        "casual", "registered", "count_real", "count_predicted",
        "residual", "absolute_error", "squared_error", "dataset_split", "model_version"
    ]
    for col in expected_columns:
        assert col in df.columns, f"Colonne requise manquante : {col}"


def test_3_residuals_and_math_consistency():
    """Test 3 : Contrôle de l'exactitude mathématique ligne à ligne des résidus."""
    df = pd.read_csv(CSV_PATH)

    # 1. residual = count_real - count_predicted
    expected_residuals = np.round(df["count_real"] - df["count_predicted"], 1)
    diff_residuals = np.abs(df["residual"] - expected_residuals)
    assert (diff_residuals < 0.05).all(), "Discordance mathématique sur residual."

    # 2. absolute_error = abs(residual)
    expected_abs = np.round(np.abs(df["residual"]), 1)
    diff_abs = np.abs(df["absolute_error"] - expected_abs)
    assert (diff_abs < 0.05).all(), "Discordance mathématique sur absolute_error."

    # 3. squared_error = residual ** 2
    expected_sq = np.round(df["residual"] ** 2, 2)
    diff_sq = np.abs(df["squared_error"] - expected_sq)
    assert (diff_sq < 0.1).all(), "Discordance mathématique sur squared_error."


def test_4_dataset_split_partition_and_positivity():
    """Test 4 : Contrôle de la répartition Train / Test et garantie physique."""
    df = pd.read_csv(CSV_PATH)

    split_counts = df["dataset_split"].value_counts().to_dict()
    assert split_counts.get("Train") == 8151, f"Effectif Train anormal : {split_counts}"
    assert split_counts.get("Test") == 2735, f"Effectif Test anormal : {split_counts}"

    # Garantie physique
    assert (df["count_predicted"] >= 0.0).all(), "Prédictions négatives détectées !"

    # Vérification que le RMSE calculé sur le sous-ensemble Test concorde avec J4 (autour de 69.4 vélos)
    test_subset = df[df["dataset_split"] == "Test"]
    test_rmse = math.sqrt(test_subset["squared_error"].mean())
    assert 68.0 < test_rmse < 71.0, f"Test RMSE hors plage attendue : {test_rmse}"
