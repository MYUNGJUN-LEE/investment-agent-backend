import sqlite3
import time
from datetime import datetime

UNIVERSE_DB = "data/universe_scanner.sqlite3"
EDGE_DB = "data/edge_calibration.sqlite3"

def count(db, query):
    try:
        con = sqlite3.connect(db)
        cur = con.cursor()
        value = cur.execute(query).fetchone()[0]
        con.close()
        return value
    except Exception as e:
        return f"ERROR: {e}"

while True:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    scanner_history = count(
        UNIVERSE_DB,
        "select count(*) from scanner_candidate_history"
    )

    scanner_candidates = count(
        UNIVERSE_DB,
        "select count(*) from scanner_candidates"
    )

    edge_samples = count(
        EDGE_DB,
        "select count(*) from edge_training_samples"
    )

    top_perf = count(
        EDGE_DB,
        "select count(*) from top_candidate_performance"
    )

    print()
    print("시간:", now)
    print("scanner_candidate_history:", scanner_history)
    print("scanner_candidates:", scanner_candidates)
    print("edge_training_samples:", edge_samples)
    print("top_candidate_performance:", top_perf)

    time.sleep(30)
