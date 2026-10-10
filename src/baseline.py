"""Module d'entraînement et d'évaluation des baselines de référence.

Calcule :
- Baseline 0 : Moyenne globale du train.
- Baseline 1 : Moyenne par (hour, workingday) apprise sur le train.
Évalue sur le jeu de test chronologique avec métriques à double niveau (RMSE, MAE, RMSLE)
et sauvegarde les résultats dans reports/baseline_metrics.json.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error

from src.split import load_data_from_db, get_temporal_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = PROJECT_ROOT / "reports"
METRICS_PATH = REPORTS_DIR / "baseline_metrics.json"


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Calcule RMSE, MAE et RMSLE entre les valeurs réelles et prédites."""
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)

    # Écrêtage strict à 0 pour éviter les valeurs négatives
    y_pred_clipped = np.clip(y_pred_arr, 0, None)

    rmse = float(np.sqrt(mean_squared_error(y_true_arr, y_pred_clipped)))
    mae = float(mean_absolute_error(y_true_arr, y_pred_clipped))
    rmsle = float(np.sqrt(np.mean((np.log1p(y_true_arr) - np.log1p(y_pred_clipped)) ** 2)))

    return {
        "rmse": round(rmse, 4),
        "mae": round(mae, 4),
        "rmsle": round(rmsle, 4),
    }


def run_baselines() -> dict:
    """Exécute l'entraînement et l'évaluation des deux baselines."""
    df = load_data_from_db()
    df["hour"] = df["datetime"].dt.hour

    train_df, test_df = get_temporal_split(df)
    y_test = test_df["count"].values

    # 1. Baseline 0 : Moyenne globale
    global_mean = float(train_df["count"].mean())
    pred_b0 = np.full(len(test_df), global_mean)
    metrics_b0 = compute_metrics(y_test, pred_b0)
    metrics_b0["model"] = "Baseline 0 (Moyenne globale train)"
    metrics_b0["global_mean"] = round(global_mean, 4)

    # 2. Baseline 1 : Moyenne par (hour, workingday)
    lookup = train_df.groupby(["hour", "workingday"])["count"].mean().to_dict()
    pred_b1 = test_df.apply(
        lambda row: lookup.get((row["hour"], row["workingday"]), global_mean),
        axis=1,
    ).values
    metrics_b1 = compute_metrics(y_test, pred_b1)
    metrics_b1["model"] = "Baseline 1 (hour x workingday)"

    # Synthèse comparative et gains mesurés
    report = {
        "dataset_split": {
            "total_rows": len(df),
            "train_rows": len(train_df),
            "train_ratio_pct": round(len(train_df) / len(df) * 100, 2),
            "train_max_date": str(train_df["datetime"].max()),
            "test_rows": len(test_df),
            "test_ratio_pct": round(len(test_df) / len(df) * 100, 2),
            "test_min_date": str(test_df["datetime"].min()),
        },
        "baseline_0_global_mean": metrics_b0,
        "baseline_1_hour_workingday": metrics_b1,
        "gain_brought_by_hour_workingday": {
            "rmse_reduction": round(metrics_b0["rmse"] - metrics_b1["rmse"], 4),
            "rmse_reduction_pct": round(
                (metrics_b0["rmse"] - metrics_b1["rmse"]) / metrics_b0["rmse"] * 100, 2
            ),
            "rmsle_reduction": round(metrics_b0["rmsle"] - metrics_b1["rmsle"], 4),
            "rmsle_reduction_pct": round(
                (metrics_b0["rmsle"] - metrics_b1["rmsle"]) / metrics_b0["rmsle"] * 100, 2
            ),
        },
    }

    # Sauvegarde du rapport JSON
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    return report


if __name__ == "__main__":
    report = run_baselines()
    print("=== RAPPORT D'ÉVALUATION DES BASELINES ===")
    print(json.dumps(report, indent=2, ensure_ascii=False))
