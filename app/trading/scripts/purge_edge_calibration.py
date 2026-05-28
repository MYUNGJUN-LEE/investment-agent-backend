from pathlib import Path
import shutil
import sqlite3

from app.config import settings


BAD_STATUSES = (
    "EXCLUDED",
    "SKIPPED",
    "ARCHIVED",
    "BLOCKED",
    "NOT_EXECUTABLE",
)


def main() -> None:
    db = Path(settings.storage_path(settings.edge_calibration_db_path))
    print(f"DB path: {db}")
    print(f"DB exists: {db.exists()}")

    if not db.exists():
        raise SystemExit("edge_calibration DB file not found")

    backup = db.with_name(db.name + ".bak")
    shutil.copy2(db, backup)
    print(f"Backup saved: {backup}")

    conn = sqlite3.connect(db)

    before = conn.execute(
        "SELECT COUNT(*) FROM edge_training_samples"
    ).fetchone()[0]

    print(f"Before total samples: {before}")
    print("Before by status:")
    for row in conn.execute(
        """
        SELECT status, COUNT(*)
        FROM edge_training_samples
        GROUP BY status
        ORDER BY COUNT(*) DESC
        """
    ):
        print(row)

    placeholders = ",".join("?" for _ in BAD_STATUSES)

    conn.execute(
        f"""
        DELETE FROM edge_training_samples
        WHERE UPPER(COALESCE(status, '')) IN ({placeholders})
        """,
        BAD_STATUSES,
    )

    conn.execute("DELETE FROM top_candidate_performance")
    conn.execute("DELETE FROM edge_calibration_runs")

    conn.commit()

    after = conn.execute(
        "SELECT COUNT(*) FROM edge_training_samples"
    ).fetchone()[0]

    print(f"After total samples: {after}")
    print(f"Deleted samples: {before - after}")
    print("After by status:")
    for row in conn.execute(
        """
        SELECT status, COUNT(*)
        FROM edge_training_samples
        GROUP BY status
        ORDER BY COUNT(*) DESC
        """
    ):
        print(row)

    conn.close()


if __name__ == "__main__":
    main()