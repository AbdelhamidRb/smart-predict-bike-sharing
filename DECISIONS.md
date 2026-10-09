# REGISTRE DES DÉCISIONS D'ARCHITECTURE & TECHNIQUES (ADR)

Ce document consigne chaque arbitrage technique structurant pris tout au long du projet Smart Predict, avec son contexte, sa justification et ses conséquences.

---

## ADR-001 : Choix du moteur de base de données relationnelle (SQLite)

- **Date** : Octobre 2026
- **Statut** : Validé
- **Décideurs** : Équipe Projet (Ingénieur & Assistant)

### Contexte
Le guide technique du projet prévoit le stockage des données nettoyées dans une base relationnelle (SQLite ou MySQL) afin de simuler un environnement de production où les flux de données proviennent d'une source SQL transactionnelle / analytique.

### Décision
Nous choisissons **SQLite** comme système de gestion de base de données (SGBD).

### Justification & Rationale Senior
1. **Zéro dépendance d'infrastructure** : SQLite est une base de données autonome stockée sous forme de fichier local. Elle ne requiert aucun serveur tiers en tâche de fond (contrairement à MySQL nécessitant XAMPP, un service Windows ou un conteneur Docker).
2. **Reproductibilité & Portabilité immédiate** : N'importe quel développeur, évaluateur ou recruteur clonant le dépôt GitHub peut exécuter l'ensemble du pipeline sans aucune étape préalable d'installation ou de configuration réseau/utilisateur.
3. **Compatibilité native Python** : SQLite est directement intégré à la bibliothèque standard Python (`sqlite3`), éliminant les risques de dérive de drivers ou de connecteurs natifs.
4. **Adéquation au volume** : Avec ~10 886 lignes de données horaires, SQLite offre des performances d'indexation et d'agrégation instantanées, largement suffisantes pour le cas d'usage.

### Conséquences
- Fichier de base de données localisé dans `data/database.db` (ou `data/processed/database.db`).
- Ce fichier de base est exclu du suivi Git via `.gitignore` pour éviter le stockage de binaires volumineux modifiés fréquemment, tandis que le script de construction SQL (`sql/schema.sql` et `src/ingestion_sql.py`) garantit sa reproductibilité déterministe.
