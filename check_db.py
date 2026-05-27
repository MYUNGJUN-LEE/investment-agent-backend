import sqlite3

db = "data/universe_scanner.sqlite3"
con = sqlite3.connect(db)
cur = con.cursor()

print("DB:", db)
print()

print("=== tables ===")
tables = cur.execute("select name from sqlite_master where type='table' order by name").fetchall()
for t in tables:
    print(t[0])

print()
print("=== scanner_candidate_history count ===")
try:
    print(cur.execute("select count(*) from scanner_candidate_history").fetchone()[0])
except Exception as e:
    print("ERROR:", e)

print()
print("=== edge_training_samples count ===")
try:
    print(cur.execute("select count(*) from edge_training_samples").fetchone()[0])
except Exception as e:
    print("ERROR:", e)

print()
print("=== scanner_candidate_history columns ===")
try:
    cols = cur.execute("pragma table_info(scanner_candidate_history)").fetchall()
    for col in cols:
        print(col[1])
except Exception as e:
    print("ERROR:", e)

print()
print("=== edge_training_samples columns ===")
try:
    cols = cur.execute("pragma table_info(edge_training_samples)").fetchall()
    for col in cols:
        print(col[1])
except Exception as e:
    print("ERROR:", e)

print()
print("=== convertible checks ===")
checks = [
    ("history_with_observed_price", "select count(*) from scanner_candidate_history where observed_price is not null"),
    ("history_with_observed_at", "select count(*) from scanner_candidate_history where observed_at is not null"),
    ("history_with_rank", "select count(*) from scanner_candidate_history where rank is not null"),
    ("convertible_candidates", "select count(*) from scanner_candidate_history where observed_price is not null and observed_at is not null and rank is not null"),
]

for name, q in checks:
    try:
        print(name, cur.execute(q).fetchone()[0])
    except Exception as e:
        print(name, "ERROR:", e)

con.close()
