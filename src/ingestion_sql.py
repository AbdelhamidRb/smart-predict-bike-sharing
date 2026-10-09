"""Script d'ingestion des données brutes Bike Sharing Demand dans SQLite.

Garantit l'atomicité, l'idempotence, le respect strict des contraintes CHECK,
la conversion ISO-8601 des timestamps et la validation transactionnelle
des 10 886 lignes attendues.
"""

from pathlib import Path
import logging
import sqlite3
import pandas as pd

# Configuration du logging structuré
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("IngestionSQL")

# Chemins structurés du projet
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = PROJECT_ROOT / "data" / "raw" / "train.csv"
SCHEMA_PATH = PROJECT_ROOT / "sql" / "schema.sql"
DB_PATH = PROJECT_ROOT / "data" / "database.db"

EXPECTED_ROW_COUNT = 10886


def load_and_prepare_data(csv_path: Path) -> pd.DataFrame:
    """Charge le fichier CSV et formate les colonnes."""
    logger.info("Chargement du fichier source : %s", csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Le fichier source {csv_path} est introuvable.")

    df = pd.read_csv(csv_path)

    # Conversion stricte en chaîne ISO-8601 YYYY-MM-DD HH:MM:SS
    df["datetime"] = pd.to_datetime(df["datetime"]).dt.strftime("%Y-%m-%d %H:%M:%S")

    # Ordonnancement strict des colonnes aligné sur le schéma DDL
    expected_cols = [
        "datetime",
        "season",
        "holiday",
        "workingday",
        "weather",
        "temp",
        "atemp",
        "humidity",
        "windspeed",
        "casual",
        "registered",
        "count",
    ]
    df = df[expected_cols]
    return df


def ingest_data():
    """Exécute l'ingestion idempotente dans SQLite sous transaction unique."""
    logger.info("Démarrage du processus d'ingestion SQL...")

    # Chargement et préparation des données en mémoire
    df = load_and_prepare_data(CSV_PATH)
    records = df.to_records(index=False).tolist()

    # Lecture du schéma DDL
    with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
        schema_sql = f.read()

    # S'assurer que le dossier cible data/ existe
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Connexion à la base de données
    conn = sqlite3.connect(DB_PATH)
    try:
        # Configuration SQLite pour lever les erreurs et activer les contraintes
        conn.isolation_level = None  # Gestion manuelle des transactions
        cursor = conn.cursor()

        # Idempotence : recréation propre de la table et des index via schema.sql (DROP TABLE IF EXISTS inclus)
        logger.info("Application du schéma DDL depuis : %s", SCHEMA_PATH)
        cursor.executescript(schema_sql)

        # Début de la transaction atomique
        cursor.execute("BEGIN TRANSACTION;")
        logger.info("Transaction ouverte. Insertion de %d lignes...", len(records))

        insert_query = """
            INSERT INTO bike_trips (
                datetime, season, holiday, workingday, weather,
                temp, atemp, humidity, windspeed, casual, registered, count
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """
        cursor.executemany(insert_query, records)

        # Contrôle d'intégrité STRICT DANS LA TRANSACTION avant tout COMMIT
        cursor.execute("SELECT COUNT(*) FROM bike_trips;")
        inserted_count = cursor.fetchone()[0]

        if inserted_count != EXPECTED_ROW_COUNT:
            logger.error(
                "Discordance de volumétrie détectée : %d lignes insérées au lieu de %d attendues !",
                inserted_count,
                EXPECTED_ROW_COUNT,
            )
            cursor.execute("ROLLBACK;")
            logger.warning("ROLLBACK exécuté. Base de données remise à l'état initial.")
            raise ValueError(
                f"Volumétrie invalide : {inserted_count} != {EXPECTED_ROW_COUNT}."
            )

        # Validation finale si le compte est rigoureusement exact
        cursor.execute("COMMIT;")
        logger.info(
            "COMMIT réussi. %d lignes insérées et validées avec intégrité dans %s.",
            inserted_count,
            DB_PATH.name,
        )

    except Exception as e:
        logger.error("Erreur critique lors de l'ingestion : %s", e)
        try:
            conn.execute("ROLLBACK;")
        except Exception:
            pass
        raise
    finally:
        conn.close()
        logger.info("Connexion SQLite fermée.")


if __name__ == "__main__":
    ingest_data()
