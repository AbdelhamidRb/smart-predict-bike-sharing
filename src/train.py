"""Script d'expérimentation, de sélection rigoureuse sur validation et d'évaluation finale.

Protocole d'ingénierie senior anti-fuite :
1. Découpage chronologique à 3 niveaux :
   - Train_fit : 01/01/2011 au 19/03/2012 (6 785 lignes, 83.2% du train)
   - Validation : 01/04/2012 au 19/06/2012 (1 366 lignes, 16.8% du train)
   - Test (gelé, juge neutre) : 01/07/2012 au 19/12/2012 (2 735 lignes)
2. Shootout systématique log1p vs sqrt sur TOUS les modèles (Random Forest et HistGradientBoosting)
   évalué STRICTEMENT sur le jeu de validation.
3. Sélection du champion uniquement sur les scores de validation.
4. Ré-entraînement du modèle champion sur l'ensemble du train (8 151 lignes)
   et ÉVALUATION UNIQUE sur le jeu de test gelé (juge neutre).
5. Sérialisation dans models/best_model.joblib et export reports/model_comparison.json.
"""

from pathlib import Path
from typing import Tuple, Dict, Any
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from src.split import load_data_from_db, get_temporal_split
from src.features import extract_features, FEATURE_COLUMNS
from src.baseline import compute_metrics

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
OUTPUT_MODEL_PATH = MODELS_DIR / "best_model.joblib"
OUTPUT_REPORT_PATH = REPORTS_DIR / "model_comparison.json"

VAL_SPLIT_THRESHOLD = "2012-04-01 00:00:00"


def evaluate_model_on_data(
    model,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_eval: pd.DataFrame,
    y_eval: np.ndarray,
    target_transform: str,
) -> Tuple[Dict[str, float], np.ndarray, object]:
    """Entraîne sur (X_train, y_train transformé) et évalue sur (X_eval, y_eval réel)."""
    # 1. Transformation de la cible
    if target_transform == "log1p":
        y_train_trans = np.log1p(y_train)
    elif target_transform == "sqrt":
        y_train_trans = np.sqrt(y_train)
    else:
        raise ValueError(f"Transformation inconnue : {target_transform}")

    # 2. Entraînement
    model.fit(X_train, y_train_trans)

    # 3. Prédiction et ré-inversion stricte vers l'échelle réelle
    preds_trans = model.predict(X_eval)
    if target_transform == "log1p":
        preds_real = np.expm1(preds_trans)
    elif target_transform == "sqrt":
        preds_real = np.square(np.clip(preds_trans, 0, None))

    # Écrêtage final de sécurité (garantie physique >= 0 vélos)
    preds_real = np.clip(preds_real, 0, None)

    # 4. Métriques sur échelle réelle
    metrics = compute_metrics(y_eval, preds_real)
    metrics["r2_real"] = round(float(r2_score(y_eval, preds_real)), 4)
    metrics["target_transform"] = target_transform

    return metrics, preds_real, model


def run_rigorous_modeling_pipeline() -> dict:
    """Exécute la sélection sur validation, ré-entraîne le champion et évalue une seule fois sur le test."""
    print("=== DÉMARRAGE DU PIPELINE DE SÉLECTION STRICTE (TRAIN / VAL / TEST) ===")

    # 1. Chargement initial
    df = load_data_from_db()
    train_full_df, test_df = get_temporal_split(df)

    # 2. Découpage interne Train_fit vs Validation
    val_threshold = pd.Timestamp(VAL_SPLIT_THRESHOLD)
    train_fit_df = train_full_df[train_full_df["datetime"] < val_threshold].copy()
    val_df = train_full_df[train_full_df["datetime"] >= val_threshold].copy()

    print(f"Train_fit  : {len(train_fit_df)} lignes (du {train_fit_df['datetime'].min()} au {train_fit_df['datetime'].max()})")
    print(f"Validation : {len(val_df)} lignes (du {val_df['datetime'].min()} au {val_df['datetime'].max()})")
    print(f"Test (gelé): {len(test_df)} lignes (du {test_df['datetime'].min()} au {test_df['datetime'].max()})")

    # Matrices de features
    X_train_fit = extract_features(train_fit_df)
    y_train_fit = train_fit_df["count"].values.astype(float)

    X_val = extract_features(val_df)
    y_val = val_df["count"].values.astype(float)

    X_train_full = extract_features(train_full_df)
    y_train_full = train_full_df["count"].values.astype(float)

    X_test = extract_features(test_df)
    y_test = test_df["count"].values.astype(float)

    # Grille de modèles et transformations à comparer sur validation
    candidate_configs = {
        "RandomForest_log1p": {
            "model": RandomForestRegressor(n_estimators=150, max_depth=16, min_samples_leaf=2, random_state=42, n_jobs=-1),
            "transform": "log1p",
        },
        "RandomForest_sqrt": {
            "model": RandomForestRegressor(n_estimators=150, max_depth=16, min_samples_leaf=2, random_state=42, n_jobs=-1),
            "transform": "sqrt",
        },
        "HistGradientBoosting_log1p": {
            "model": HistGradientBoostingRegressor(max_iter=200, max_depth=10, min_samples_leaf=15, l2_regularization=0.1, random_state=42),
            "transform": "log1p",
        },
        "HistGradientBoosting_sqrt": {
            "model": HistGradientBoostingRegressor(max_iter=200, max_depth=10, min_samples_leaf=15, l2_regularization=0.1, random_state=42),
            "transform": "sqrt",
        },
    }

    # 3. Évaluation stricte sur VALIDATION
    print("\n--- PHASE 1 : SÉLECTION DU CHAMPION SUR LE JEU DE VALIDATION ---")
    val_results = {}
    for name, config in candidate_configs.items():
        m_val, _, _ = evaluate_model_on_data(
            config["model"], X_train_fit, y_train_fit, X_val, y_val, target_transform=config["transform"]
        )
        val_results[name] = m_val
        print(f"Validation [{name}] RMSE: {m_val['rmse']} | RMSLE: {m_val['rmsle']} | R²: {m_val['r2_real']}")

    # Sélection du champion sur validation (critère RMSLE prioritaire, RMSE en départage)
    champion_name = min(val_results.keys(), key=lambda k: val_results[k]["rmsle"])
    champion_val_metrics = val_results[champion_name]
    champion_config = candidate_configs[champion_name]

    print(f"\n=> CHAMPION SÉLECTIONNÉ SUR VALIDATION : {champion_name}")
    print(f"   Validation RMSLE: {champion_val_metrics['rmsle']} | Validation RMSE: {champion_val_metrics['rmse']}")

    # 4. PHASE 2 : Ré-entraînement du champion sur le Train complet (8 151 lignes)
    print("\n--- PHASE 2 : RÉ-ENTRAÎNEMENT DU CHAMPION SUR TRAIN COMPLET (8 151 LIGNES) ---")
    # Instanciation d'un estimateur neuf avec les mêmes hyperparamètres
    if "HistGradientBoosting" in champion_name:
        final_estimator = HistGradientBoostingRegressor(
            max_iter=200, max_depth=10, min_samples_leaf=15, l2_regularization=0.1, random_state=42
        )
    else:
        final_estimator = RandomForestRegressor(
            n_estimators=150, max_depth=16, min_samples_leaf=2, random_state=42, n_jobs=-1
        )

    # 5. PHASE 3 : Évaluation UNIQUE sur le jeu de test gelé (juge neutre)
    print("\n--- PHASE 3 : ÉVALUATION UNIQUE DU CHAMPION SUR TEST GELÉ (JUGE NEUTRE) ---")
    test_metrics, _, fitted_final = evaluate_model_on_data(
        final_estimator, X_train_full, y_train_full, X_test, y_test, target_transform=champion_config["transform"]
    )
    print(f"Test final [{champion_name}] RMSE: {test_metrics['rmse']} vélos | RMSLE: {test_metrics['rmsle']} | R²: {test_metrics['r2_real']}")

    # Contrats de performance de J3 (Baseline 1 : RMSE 144.4269, RMSLE 0.5304)
    baseline_thresholds = {
        "rmse_contract": 144.4269,
        "rmsle_contract": 0.5304,
    }

    # 6. Sauvegarde du modèle champion sérialisé
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib_payload = {
        "model_name": champion_name,
        "estimator": fitted_final,
        "feature_columns": FEATURE_COLUMNS,
        "target_transform": champion_config["transform"],
        "metrics_validation": champion_val_metrics,
        "metrics_test": test_metrics,
    }
    joblib.dump(joblib_payload, OUTPUT_MODEL_PATH)
    print(f"Modèle champion sauvegardé dans : {OUTPUT_MODEL_PATH}")

    # 7. Rapport JSON complet
    final_report = {
        "split_protocol": {
            "train_fit_rows": len(train_fit_df),
            "train_fit_max_date": str(train_fit_df["datetime"].max()),
            "val_rows": len(val_df),
            "val_min_date": str(val_df["datetime"].min()),
            "val_max_date": str(val_df["datetime"].max()),
            "test_rows": len(test_df),
            "test_min_date": str(test_df["datetime"].min()),
            "test_max_date": str(test_df["datetime"].max()),
        },
        "performance_contract_baseline_1": baseline_thresholds,
        "validation_selection_shootout": val_results,
        "champion_selected_on_validation": {
            "name": champion_name,
            "validation_metrics": champion_val_metrics,
            "test_metrics_neutral_judge": test_metrics,
            "gain_vs_baseline_1_on_test": {
                "rmse_reduction": round(baseline_thresholds["rmse_contract"] - test_metrics["rmse"], 4),
                "rmse_reduction_pct": round(
                    (baseline_thresholds["rmse_contract"] - test_metrics["rmse"])
                    / baseline_thresholds["rmse_contract"]
                    * 100,
                    2,
                ),
                "rmsle_reduction": round(baseline_thresholds["rmsle_contract"] - test_metrics["rmsle"], 4),
                "rmsle_reduction_pct": round(
                    (baseline_thresholds["rmsle_contract"] - test_metrics["rmsle"])
                    / baseline_thresholds["rmsle_contract"]
                    * 100,
                    2,
                ),
            },
        },
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2, ensure_ascii=False)
    print(f"Rapport enregistré dans : {OUTPUT_REPORT_PATH}")

    return final_report


if __name__ == "__main__":
    run_rigorous_modeling_pipeline()
