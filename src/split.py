"""Module de chargement et découpage temporel des données depuis SQLite.

Garantit :
- L'accès en lecture seule stricte à la base (mode=ro).
- L'absence formelle de shuffle aléatoire.
- Un découpage chronologique déterministe au 01/07/2012 00:00:00.
- La vérification stricte de l'étanchéité temporelle.
"""

from pathlib import Path
from typing import Tuple
import sqlite3
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "data" / "database.db"

# Date de coupure chronologique
SPLIT_DATE_THRESHOLD = "2012-07-01 00:00:00"


def load_data_from_db(db_path: Path = DB_PATH) -> pd.DataFrame:
    """Charge les données depuis la base SQLite en mode lecture seule."""
    if not db_path.exists():
        raise FileNotFoundError(f"Base de données introuvable : {db_path}")

    uri = f"{db_path.resolve().as_uri()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    try:
        query = """
            SELECT 
                datetime,
                season,
                holiday,
                workingday,
                weather,
                temp,
                atemp,
                humidity,
                windspeed,
                casual,
                registered,
                "count"
            FROM bike_trips
            ORDER BY datetime ASC;
        """
        df = pd.read_sql_query(query, conn)
    finally:
        conn.close()

    df["datetime"] = pd.to_datetime(df["datetime"])
    return df


def get_temporal_split(
    df: pd.DataFrame, split_date: str = SPLIT_DATE_THRESHOLD
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Découpe chronologiquement le DataFrame selon la date pivot.

    Train : strictement avant split_date (jusqu'au 19/06/2012 23:00:00).
    Test  : à partir de split_date (du 01/07/2012 00:00:00 au 19/12/2012 23:00:00).
    """
    threshold = pd.Timestamp(split_date)
    train_df = df[df["datetime"] < threshold].copy()
    test_df = df[df["datetime"] >= threshold].copy()

    # Contrôles stricts d'étanchéité temporelle
    assert not train_df.empty, "Le jeu d'entraînement est vide."
    assert not test_df.empty, "Le jeu de test est vide."
    assert train_df["datetime"].max() < test_df["datetime"].min(), (
        f"Fuite temporelle détectée : train max ({train_df['datetime'].max()}) "
        f">= test min ({test_df['datetime'].min()})"
    )

    return train_df, test_df


if __name__ == "__main__":
    df = load_data_from_db()
    train, test = get_temporal_split(df)
    print(f"Chargement terminé : Total={len(df)} lignes.")
    print(f"Train : {len(train)} lignes ({len(train)/len(df)*100:.2f}%) | Max date : {train['datetime'].max()}")
    print(f"Test  : {len(test)} lignes ({len(test)/len(df)*100:.2f}%) | Min date : {test['datetime'].min()}")
