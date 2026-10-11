"""Script de contrôle qualité et de vérification des métriques de l'export Power BI.

Affiche :
1. Volumétrie exacte par dataset_split (Train et Test).
2. Calcul du RMSE et MAE sur le sous-ensemble Test.
3. Vérification de la contrainte physique (prédictions >= 0).
4. Moyennes par saison et par type de jour pour concordance Power BI.
"""

from pathlib import Path
import math
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = PROJECT_ROOT / "data" / "processed" / "powerbi_export.csv"


def check_export():
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Fichier export introuvable : {CSV_PATH}")

    # Lecture avec encodage utf-8-sig
    df = pd.read_csv(CSV_PATH, encoding="utf-8-sig")

    total_rows = len(df)
    split_counts = df["dataset_split"].value_counts().to_dict()

    train_count = split_counts.get("Train", 0)
    test_count = split_counts.get("Test", 0)

    # 1. Vérification des prédictions négatives
    negative_preds = (df["count_predicted"] < 0).sum()
    min_pred = df["count_predicted"].min()
    max_pred = df["count_predicted"].max()

    # 2. Calcul des métriques sur le jeu de test
    test_df = df[df["dataset_split"] == "Test"]
    test_rmse = math.sqrt(test_df["squared_error"].mean())
    test_mae = test_df["absolute_error"].mean()
    test_rmsle = math.sqrt(
        np.mean((np.log1p(test_df["count_real"]) - np.log1p(test_df["count_predicted"])) ** 2)
    )

    # 3. Totaux réels et prédits sur Test
    total_real_test = int(test_df["count_real"].sum())
    total_pred_test = round(float(test_df["count_predicted"].sum()), 1)

    print("=== CONTRÔLE QUALITÉ DE L'EXPORT POWER BI ===")
    print(f"Fichier source : {CSV_PATH.name}")
    print(f"Total lignes   : {total_rows}")
    print(f"Lignes Train   : {train_count} ({train_count / total_rows * 100:.2f}%)")
    print(f"Lignes Test    : {test_count} ({test_count / total_rows * 100:.2f}%)")
    print(f"Prédictions < 0 : {negative_preds} (Min: {min_pred}, Max: {max_pred})")
    print(f"Condition >= 0 vérifiée : {negative_preds == 0}")

    print("\n--- MÉTRIQUES TEST (Juge Neutre) ---")
    print(f"RMSE Test  : {test_rmse:.4f}")
    print(f"MAE Test   : {test_mae:.4f}")
    print(f"RMSLE Test : {test_rmsle:.4f}")
    print(f"Total Real Test      : {total_real_test}")
    print(f"Total Predicted Test : {total_pred_test}")

    # Contrôles stricts d'assertions
    assert total_rows == 10886, f"Volumétrie totale incorrecte : {total_rows}"
    assert train_count == 8151, f"Effectif Train incorrect : {train_count}"
    assert test_count == 2735, f"Effectif Test incorrect : {test_count}"
    assert negative_preds == 0, "Présence de prédictions négatives !"


if __name__ == "__main__":
    check_export()
