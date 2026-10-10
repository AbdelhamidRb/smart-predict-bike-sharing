"""Tests automatisés pour la baseline temporelle et les métriques d'évaluation.

Vérifie :
1. Calcul mathématique des métriques sur un exemple miniature vérifié à la main.
2. Étanchéité temporelle stricte et volumétrie du split chronologique.
3. Supériorité mesurée de Baseline 1 (hour x workingday) sur Baseline 0 (globale).
"""

from pathlib import Path
import math
import numpy as np
import pandas as pd
import pytest

from src.split import load_data_from_db, get_temporal_split
from src.baseline import compute_metrics, run_baselines

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_1_metrics_calculation_handcrafted():
    """Test 1 : Vérifie compute_metrics sur un cas miniature calculé à la main.

    Données :
      y_true = [10, 20, 30]
      y_pred = [12, 18, 33]

    Calculs à la main :
      MAE = (|10-12| + |20-18| + |30-33|) / 3 = (2 + 2 + 3) / 3 = 7/3 ≈ 2.3333
      MSE = ((10-12)^2 + (20-18)^2 + (30-33)^2) / 3 = (4 + 4 + 9) / 3 = 17/3 ≈ 5.6667
      RMSE = sqrt(17/3) ≈ 2.3805
      RMSLE = sqrt(1/3 * [(ln 11 - ln 13)^2 + (ln 21 - ln 19)^2 + (ln 31 - ln 34)^2])
            ≈ sqrt(1/3 * [(-0.16705)^2 + (0.10008)^2 + (-0.09237)^2])
            ≈ sqrt(1/3 * [0.027907 + 0.010017 + 0.008533])
            ≈ sqrt(0.016152) ≈ 0.1271
    """
    y_true = np.array([10, 20, 30])
    y_pred = np.array([12, 18, 33])

    metrics = compute_metrics(y_true, y_pred)

    expected_mae = 7.0 / 3.0
    expected_rmse = math.sqrt(17.0 / 3.0)
    expected_rmsle = math.sqrt(
        (
            (math.log(11) - math.log(13)) ** 2
            + (math.log(21) - math.log(19)) ** 2
            + (math.log(31) - math.log(34)) ** 2
        )
        / 3.0
    )

    assert math.isclose(metrics["mae"], expected_mae, abs_tol=0.0001), (
        f"MAE calculé {metrics['mae']} != attendu {expected_mae}"
    )
    assert math.isclose(metrics["rmse"], expected_rmse, abs_tol=0.0001), (
        f"RMSE calculé {metrics['rmse']} != attendu {expected_rmse}"
    )
    assert math.isclose(metrics["rmsle"], expected_rmsle, abs_tol=0.0001), (
        f"RMSLE calculé {metrics['rmsle']} != attendu {expected_rmsle}"
    )


def test_2_temporal_split_boundaries_and_volume():
    """Test 2 : Contrôle rigoureux de la frontière temporelle et des volumes.

    Attentes mesurées sur la base :
      - Total = 10 886 lignes
      - Train = 8 151 lignes (74.88%)
      - Test  = 2 735 lignes (25.12%)
      - Train max date = 2012-06-19 23:00:00 (Kaggle s'arrête au 19)
      - Test min date  = 2012-07-01 00:00:00
      - Trou calendaire exact : du 20/06/2012 au 30/06/2012 (aucune fuite)
    """
    df = load_data_from_db()
    train_df, test_df = get_temporal_split(df)

    assert len(df) == 10886, f"Total lignes {len(df)} != 10886"
    assert len(train_df) == 8151, f"Train lignes {len(train_df)} != 8151"
    assert len(test_df) == 2735, f"Test lignes {len(test_df)} != 2735"

    train_max = train_df["datetime"].max()
    test_min = test_df["datetime"].min()

    # Frontière d'étanchéité stricte
    assert train_max == pd.Timestamp("2012-06-19 23:00:00"), (
        f"Train max date {train_max} inattendue"
    )
    assert test_min == pd.Timestamp("2012-07-01 00:00:00"), (
        f"Test min date {test_min} inattendue"
    )
    assert train_max < test_min, "Violation de l'étanchéité temporelle !"


def test_3_baseline_gain_and_reproducibility():
    """Test 3 : Vérifie la supériorité de Baseline 1 sur Baseline 0 et les scores exacts.

    Baseline 0 (moyenne globale = 167.7858) :
      RMSE ≈ 235.98, MAE ≈ 178.28, RMSLE ≈ 1.3598
    Baseline 1 (hour x workingday) :
      RMSE ≈ 144.43, MAE ≈ 102.17, RMSLE ≈ 0.5304

    Gain vérifié :
      - Gain RMSLE > 0.80 (> 60% de réduction d'erreur relative)
      - Gain RMSE > 90 vélos (> 38% de réduction d'erreur quadratique)
    """
    report = run_baselines()

    b0 = report["baseline_0_global_mean"]
    b1 = report["baseline_1_hour_workingday"]

    # Concordance des valeurs calculées
    assert math.isclose(b0["rmse"], 235.9797, abs_tol=0.01)
    assert math.isclose(b0["rmsle"], 1.3598, abs_tol=0.01)

    assert math.isclose(b1["rmse"], 144.4269, abs_tol=0.01)
    assert math.isclose(b1["rmsle"], 0.5304, abs_tol=0.01)

    # Preuve du gain de la baseline temporelle
    assert b1["rmse"] < b0["rmse"], "Baseline 1 ne surpasse pas Baseline 0 en RMSE."
    assert b1["rmsle"] < b0["rmsle"], "Baseline 1 ne surpasse pas Baseline 0 en RMSLE."

    gain = report["gain_brought_by_hour_workingday"]
    assert gain["rmse_reduction_pct"] > 35.0, "Gain RMSE insuffisant"
    assert gain["rmsle_reduction_pct"] > 55.0, "Gain RMSLE insuffisant"
