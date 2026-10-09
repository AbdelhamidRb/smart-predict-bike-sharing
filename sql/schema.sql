-- Schema DDL pour la base de données relationnelle du projet Smart Predict
-- SGBD cible : SQLite

DROP TABLE IF EXISTS bike_trips;

CREATE TABLE bike_trips (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    datetime TEXT NOT NULL UNIQUE,
    season INTEGER NOT NULL CHECK (season BETWEEN 1 AND 4),
    holiday INTEGER NOT NULL CHECK (holiday IN (0, 1)),
    workingday INTEGER NOT NULL CHECK (workingday IN (0, 1)),
    weather INTEGER NOT NULL CHECK (weather BETWEEN 1 AND 4),
    temp REAL NOT NULL CHECK (temp >= -20.0 AND temp <= 60.0),
    atemp REAL NOT NULL CHECK (atemp >= -25.0 AND atemp <= 65.0),
    humidity REAL NOT NULL CHECK (humidity >= 0.0 AND humidity <= 100.0),
    windspeed REAL NOT NULL CHECK (windspeed >= 0.0 AND windspeed <= 150.0),
    casual INTEGER NOT NULL CHECK (casual >= 0),
    registered INTEGER NOT NULL CHECK (registered >= 0),
    count INTEGER NOT NULL CHECK (count >= 0),
    CONSTRAINT chk_total_consistency CHECK (casual + registered = count)
);

-- Index pour optimiser les requêtes chronologiques du pipeline ML
CREATE INDEX idx_bike_trips_datetime ON bike_trips(datetime);
