"""Module de Feature Engineering anti-fuite temporelle.

Règles d'ingénierie :
- Exclusion formelle de casual et registered (ADR-003 : anti-data leakage).
- Exclusion de atemp pour supprimer la colinéarité extrême avec temp (ADR-004).
- Encodage des composantes temporelles cycliques (sin/cos pour l'heure et le mois).
- Ajout de l'année (tendance structurelle 2011 vs 2012) et du jour de la semaine (0 à 6).
- Indicateur binaire windspeed_is_zero pour documenter l'anomalie de capteur (ADR-005).
"""

from typing import Tuple, List
import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    # Calendrier et temporalité linéaire
    "year",
    "workingday",
    "holiday",
    "dayofweek",
    # Temporalité cyclique (préservation de la continuité 23h -> 0h et déc -> janv)
    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos",
    # Météo
    "season",
    "weather",
    "temp",
    "humidity",
    "windspeed",
    "windspeed_is_zero",
]


def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """Transforme un DataFrame brut extrait de la base SQL en matrice de features."""
    X = pd.DataFrame(index=df.index)

    # Conversion datetime si nécessaire
    dt = pd.to_datetime(df["datetime"])

    # 1. Variables temporelles
    hour = dt.dt.hour
    month = dt.dt.month
    X["year"] = dt.dt.year
    X["workingday"] = df["workingday"].astype(int)
    X["holiday"] = df["holiday"].astype(int)
    X["dayofweek"] = dt.dt.dayofweek

    # Encodage sin / cos cyclique
    X["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    X["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    X["month_sin"] = np.sin(2 * np.pi * (month - 1) / 12.0)
    X["month_cos"] = np.cos(2 * np.pi * (month - 1) / 12.0)

    # 2. Variables environnementales & météo
    X["season"] = df["season"].astype(int)
    X["weather"] = df["weather"].astype(int)
    X["temp"] = df["temp"].astype(float)
    X["humidity"] = df["humidity"].astype(float)
    X["windspeed"] = df["windspeed"].astype(float)

    # Indicateur binaire pour les zéros anémomètre (ADR-005)
    X["windspeed_is_zero"] = (df["windspeed"] == 0.0).astype(int)

    # Vérification stricte de l'absence de colonnes interdites
    forbidden = {"casual", "registered", "atemp", "count"}
    assert forbidden.isdisjoint(X.columns), f"Fuite de données : {forbidden & set(X.columns)}"

    return X[FEATURE_COLUMNS]


def prepare_datasets(
    train_df: pd.DataFrame, test_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """Extrait les matrices de features X_train, X_test et les cibles réelles y_train, y_test."""
    X_train = extract_features(train_df)
    X_test = extract_features(test_df)

    y_train = train_df["count"].values.astype(float)
    y_test = test_df["count"].values.astype(float)

    return X_train, X_test, y_train, y_test
