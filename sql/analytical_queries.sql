-- =============================================================================
-- REQUÊTES ANALYTIQUES DE VALIDATION ET D'INTÉGRITÉ (Projet Smart Predict)
-- SGBD : SQLite
-- Base cible : data/database.db
-- =============================================================================

-- -----------------------------------------------------------------------------
-- Q1 : Contrôle global de volumétrie et détection exhaustive de valeurs NULL
-- Objectif attendu : total_rows = 10886, unique_datetimes = 10886, total_nulls = 0
-- -----------------------------------------------------------------------------
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


-- -----------------------------------------------------------------------------
-- Q2 : Contrôle d'intégrité de la décomposition de la cible
-- Règle métier : casual + registered = "count"
-- Résultat attendu : 0 ligne discordante
-- -----------------------------------------------------------------------------
SELECT 
    COUNT(*) AS mismatch_count
FROM bike_trips
WHERE casual + registered != "count";


-- -----------------------------------------------------------------------------
-- Q3 : Moyenne des locations et effectifs par saison
-- Résultats attendus :
--   - Printemps (1) : 116.34 vélos/h
--   - Été (2)       : 215.25 vélos/h
--   - Automne (3)   : 234.42 vélos/h
--   - Hiver (4)     : 198.99 vélos/h
-- -----------------------------------------------------------------------------
SELECT 
    season,
    CASE season
        WHEN 1 THEN '1: Printemps'
        WHEN 2 THEN '2: Été'
        WHEN 3 THEN '3: Automne'
        WHEN 4 THEN '4: Hiver'
    END AS season_label,
    ROUND(AVG("count"), 2) AS avg_count,
    COUNT(*) AS observation_count
FROM bike_trips
GROUP BY season
ORDER BY season;


-- -----------------------------------------------------------------------------
-- Q4 : Moyenne des locations et effectifs par condition météorologique
-- Résultats attendus :
--   - Beau (1)                : 205.24 vélos/h (7 192 obs)
--   - Brouillard/Nuages (2)   : 178.96 vélos/h (2 834 obs)
--   - Neige/Pluie légère (3)  : 118.85 vélos/h (859 obs)
--   - Pluie forte/Orage (4)   : N = 1 obs (moyenne 164.00 non représentative)
-- -----------------------------------------------------------------------------
SELECT 
    weather,
    CASE weather
        WHEN 1 THEN '1: Beau'
        WHEN 2 THEN '2: Brouillard/Nuages'
        WHEN 3 THEN '3: Neige/Pluie légère'
        WHEN 4 THEN '4: Pluie forte/Orage'
    END AS weather_label,
    COUNT(*) AS observation_count,
    ROUND(AVG("count"), 2) AS avg_count
FROM bike_trips
GROUP BY weather
ORDER BY weather;


-- -----------------------------------------------------------------------------
-- Q5 : Profil pendulaire aux heures de pointe (8h et 17h) selon workingday
-- Résultats attendus (4 lignes) :
--   - 8h  | Non-ouvré (0) : 112.26 | Ouvré (1) : 479.95
--   - 17h | Non-ouvré (0) : 339.12 | Ouvré (1) : 529.21
-- -----------------------------------------------------------------------------
SELECT 
    CAST(strftime('%H', datetime) AS INTEGER) AS hour_of_day,
    workingday,
    ROUND(AVG("count"), 2) AS avg_count,
    COUNT(*) AS observation_count
FROM bike_trips
WHERE CAST(strftime('%H', datetime) AS INTEGER) IN (8, 17)
GROUP BY hour_of_day, workingday
ORDER BY hour_of_day, workingday;
