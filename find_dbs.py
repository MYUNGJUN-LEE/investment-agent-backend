import sqlite3
from pathlib import Path

for p in list(Path(".").rglob("*.sqlite3")) + list(Path(".").rglob("*.sqlite")) + list(Path(".").rglob("*.db")):
    try:
        con = sqlite3.connect(str(p))
        cur = con.cursor()
        tables = [x[0] for x in cur.execute("select name from sqlite_master where type='table'").fetchall()]
        con.close()

        if "edge_training_samples" in tables or "scanner_candidate_history" in tables:
            print()
            print("DB:", p)
            print("tables:", tables)

    except Exception as e:
        pass
