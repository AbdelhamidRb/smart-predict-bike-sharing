"""Suite de tests automatisés d'intégrité et de cohérence pour la base SQL SQLite.

Exécute les 6 tests de validation numérique et structurelle
alignés sur l'EDA de J1 et le schéma DDL de J2.
Connexion en lecture seule stricte (mode=ro).
"""

from pathlib import Path
import math
import sqlite3

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "data" / "database.db"


def get_db_connection() -> sqlite3.Connection:
    """Retourne une connexion en lecture seule stricte à la base SQLite."""
    assert DB_PATH.exists(), f"La base de données {DB_PATH} n'existe pas."
    # Connexion en mode lecture seule stricte (mode=ro)
    uri = f"{DB_PATH.resolve().as_uri()}?mode=ro"
    return sqlite3.connect(uri, uri=True)


def test_1_table_and_indexes():
    """Test 1 : Vérifie la présence de la table bike_trips et de l'index B-tree."""
    conn = get_db_connection()
    cursor = conn.cursor()

    table = cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='bike_trips';"
    ).fetchone()
    assert table is not None, "La table bike_trips est absente de sqlite_master."

    indexes = [
        row[0]
        for row in cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='bike_trips';"
        ).fetchall()
    ]
    conn.close()

    assert "idx_bike_trips_datetime" in indexes, (
        f"L'index B-tree idx_bike_trips_datetime est manquant. Index trouvés: {indexes}"
    )


def test_2_row_count_and_nulls():
    """Test 2 : Contrôle volumétrique (10 886) et absence absolue de NULL."""
    conn = get_db_connection()
    cursor = conn.cursor()

    q1 = """
    SELECT 
        COUNT(*) AS total_rows,
        COUNT(DISTINCT datetime) AS unique_datetimes,
        SUM(
            CASE WHEN id IS NULL THEN 1 ELSE 0 END +
            CASE WHEN datetime IS NULL THEN 1 ELSE 0 END +
            CASE WHEN season IS NULL THEN 1 ELSE 0 END +
            CASE WHEN holiday IS NULL THEN 1 ELSE 0 END +
            CASE WHEN workingday IS NULL THEN 1 ELSE 0 END +
            CASE WHEN weather IS NULL THEN 1 ELSE 0 END +
            CASE WHEN temp IS NULL THEN 1 ELSE 0 END +
            CASE WHEN atemp IS NULL THEN 1 ELSE 0 END +
            CASE WHEN humidity IS NULL THEN 1 ELSE 0 END +
            CASE WHEN windspeed IS NULL THEN 1 ELSE 0 END +
            CASE WHEN casual IS NULL THEN 1 ELSE 0 END +
            CASE WHEN registered IS NULL THEN 1 ELSE 0 END +
            CASE WHEN "count" IS NULL THEN 1 ELSE 0 END
        ) AS total_null_occurrences
    FROM bike_trips;
    """
    total_rows, unique_datetimes, total_nulls = cursor.execute(q1).fetchone()
    conn.close()

    assert total_rows == 10886, f"Volumétrie incorrecte : {total_rows} != 10886"
    assert unique_datetimes == 10886, (
        f"Doublons d'horodatage détectés : {unique_datetimes} != 10886"
    )
    assert total_nulls == 0, f"Des valeurs NULL ont été détectées : {total_nulls}"


def test_3_target_composition():
    """Test 3 : Contrôle de l'égalité stricte casual + registered = count."""
    conn = get_db_connection()
    cursor = conn.cursor()

    q2 = """
    SELECT COUNT(*) AS mismatch_count
    FROM bike_trips
    WHERE casual + registered != "count";
    """
    mismatches = cursor.execute(q2).fetchone()[0]
    conn.close()

    assert mismatches == 0, (
        f"Discordance de décomposition de count : {mismatches} lignes incohérentes."
    )


def test_4_season_averages():
    """Test 4 : Moyenne des locations par saison (tolérance ±0.02)."""
    conn = get_db_connection()
    cursor = conn.cursor()

    q3 = """
    SELECT season, ROUND(AVG("count"), 2) AS avg_count, COUNT(*) AS n_obs
    FROM bike_trips
    GROUP BY season
    ORDER BY season;
    """
    rows = cursor.execute(q3).fetchall()
    conn.close()

    assert len(rows) == 4, f"4 saisons attendues, reçu : {len(rows)}"

    expected_averages = {
        1: 116.34,  # Printemps
        2: 215.25,  # Été
        3: 234.42,  # Automne
        4: 198.99,  # Hiver
    }

    for season, avg_count, n_obs in rows:
        target = expected_averages[season]
        assert math.isclose(avg_count, target, abs_tol=0.02), (
            f"Saison {season} : moyenne {avg_count} éloignée de la cible {target}"
        )
        assert n_obs > 2600, (
            f"Saison {season} : effectif anormalement bas ({n_obs})"
        )


def test_5_weather_statistics():
    """Test 5 : Effectifs et moyennes par condition météo (Météo 4 : effectif strict N=1)."""
    conn = get_db_connection()
    cursor = conn.cursor()

    q4 = """
    SELECT weather, COUNT(*) AS n_obs, ROUND(AVG("count"), 2) AS avg_count
    FROM bike_trips
    GROUP BY weather
    ORDER BY weather;
    """
    rows = cursor.execute(q4).fetchall()
    conn.close()

    assert len(rows) == 4, f"4 catégories météo attendues, reçu : {len(rows)}"

    weather_data = {row[0]: (row[1], row[2]) for row in rows}

    # Weather 1 (Beau)
    assert weather_data[1][0] == 7192, (
        f"Weather 1 : N={weather_data[1][0]} != 7192"
    )
    assert math.isclose(weather_data[1][1], 205.24, abs_tol=0.02), (
        f"Weather 1 : moyenne {weather_data[1][1]} != 205.24"
    )

    # Weather 2 (Nuages/Brouillard)
    assert weather_data[2][0] == 2834, (
        f"Weather 2 : N={weather_data[2][0]} != 2834"
    )
    assert math.isclose(weather_data[2][1], 178.96, abs_tol=0.02), (
        f"Weather 2 : moyenne {weather_data[2][1]} != 178.96"
    )

    # Weather 3 (Neige/Pluie légère)
    assert weather_data[3][0] == 859, (
        f"Weather 3 : N={weather_data[3][0]} != 859"
    )
    assert math.isclose(weather_data[3][1], 118.85, abs_tol=0.02), (
        f"Weather 3 : moyenne {weather_data[3][1]} != 118.85"
    )

    # Weather 4 (Pluie forte/Orage) : validation stricte du volume (1 observation)
    assert weather_data[4][0] == 1, (
        f"Weather 4 : N={weather_data[4][0]} != 1 observation attendue."
    )


def test_6_commute_peaks():
    """Test 6 : Profil pendulaire 8h et 17h selon workingday (4 segments)."""
    conn = get_db_connection()
    cursor = conn.cursor()

    q5 = """
    SELECT 
        CAST(strftime('%H', datetime) AS INTEGER) AS hour_of_day,
        workingday,
        ROUND(AVG("count"), 2) AS avg_count,
        COUNT(*) AS observation_count
    FROM bike_trips
    WHERE CAST(strftime('%H', datetime) AS INTEGER) IN (8, 17)
    GROUP BY hour_of_day, workingday
    ORDER BY hour_of_day, workingday;
    """
    rows = cursor.execute(q5).fetchall()
    conn.close()

    assert len(rows) == 4, (
        f"4 segments horaires attendus (8h/17h x workingday 0/1), reçu : {len(rows)}"
    )

    expected = {
        (8, 0): (112.26, 145),
        (8, 1): (479.95, 310),
        (17, 0): (339.12, 145),
        (17, 1): (529.21, 311),
    }

    for hour, work, avg_count, n_obs in rows:
        target_avg, target_n = expected[(hour, work)]
        assert math.isclose(avg_count, target_avg, abs_tol=0.02), (
            f"Heure {hour} (workingday={work}) : moyenne {avg_count} != {target_avg}"
        )
        assert n_obs == target_n, (
            f"Heure {hour} (workingday={work}) : effectif {n_obs} != {target_n}"
        )
