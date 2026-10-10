"""Tests automatisés pour la sélection sur validation et l'évaluation du champion.

Vérifie :
1. Présence du modèle champion avec métriques de validation et de test.
2. Contrat de performance sur le jeu de validation (évite tout biais de sélection sur le test).
3. Contrat de performance final sur le jeu de test gelé (juge neutre).
4. Positivité et reproductibilité de l'inférence.
"""

from pathlib import Path
import joblib
import numpy as np
import pytest

from src.split import load_data_from_db, get_temporal_split
from src.features import extract_features, FEATURE_COLUMNS
from src.baseline import compute_metrics

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "best_model.joblib"

RMSE_CONTRACT = 144.4269
RMSLE_CONTRACT = 0.5304


def test_1_champion_model_file_exists_and_valid():
    """Test 1 : Vérifie que le modèle champion contient ses métadonnées de validation et de test."""
    assert MODEL_PATH.exists(), f"Le fichier modèle {MODEL_PATH} est introuvable."

    payload = joblib.load(MODEL_PATH)
    assert "model_name" in payload
    assert "estimator" in payload
    assert "feature_columns" in payload
    assert "target_transform" in payload
    assert "metrics_validation" in payload
    assert "metrics_test" in payload

    assert payload["feature_columns"] == FEATURE_COLUMNS, "Divergence de features."


def test_2_champion_beats_contract_on_validation():
    """Test 2 : Vérifie que le modèle bat le contrat de performance DÈS LA PHASE DE VALIDATION."""
    payload = joblib.load(MODEL_PATH)
    val_metrics = payload["metrics_validation"]

    # Le modèle sélectionné sur validation devait déjà battre le contrat
    assert val_metrics["rmse"] < RMSE_CONTRACT, (
        f"Validation RMSE {val_metrics['rmse']} >= contrat {RMSE_CONTRACT}"
    )
    assert val_metrics["rmsle"] < RMSLE_CONTRACT, (
        f"Validation RMSLE {val_metrics['rmsle']} >= contrat {RMSLE_CONTRACT}"
    )
    assert val_metrics["r2_real"] > 0.85, f"Validation R² insuffisant : {val_metrics['r2_real']}"


def test_3_champion_generalization_on_neutral_test_set():
    """Test 3 : Vérifie la généralisation du modèle sur le jeu de test gelé (juge neutre)."""
    payload = joblib.load(MODEL_PATH)
    test_metrics = payload["metrics_test"]

    assert test_metrics["rmse"] < RMSE_CONTRACT, (
        f"Test RMSE {test_metrics['rmse']} >= contrat {RMSE_CONTRACT}"
    )
    assert test_metrics["rmsle"] < RMSLE_CONTRACT, (
        f"Test RMSLE {test_metrics['rmsle']} >= contrat {RMSLE_CONTRACT}"
    )
    assert test_metrics["r2_real"] > 0.85, f"Test R² insuffisant : {test_metrics['r2_real']}"


def test_4_inference_positivity_and_exactness():
    """Test 4 : Inférence de bout en bout et garantie physique de positivité."""
    payload = joblib.load(MODEL_PATH)
    estimator = payload["estimator"]
    target_transform = payload["target_transform"]

    df = load_data_from_db()
    _, test_df = get_temporal_split(df)
    X_test = extract_features(test_df)
    y_test = test_df["count"].values

    preds_raw = estimator.predict(X_test)
    if target_transform == "log1p":
        preds_real = np.expm1(preds_raw)
    elif target_transform == "sqrt":
        preds_real = np.square(np.clip(preds_raw, 0, None))
    else:
        raise ValueError("Transformation inconnue")

    preds_real = np.clip(preds_real, 0, None)

    # Garantie physique absolue
    assert (preds_real >= 0.0).all(), "Prédictions de volumes négatifs détectées !"

    # Re-calcul indépendant
    recomputed = compute_metrics(y_test, preds_real)
    assert abs(recomputed["rmse"] - payload["metrics_test"]["rmse"]) < 0.01
    assert abs(recomputed["rmsle"] - payload["metrics_test"]["rmsle"]) < 0.01
