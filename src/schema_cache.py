"""
schema_cache.py — Phase-A schema compiler.

Given a .sqlite path, returns a dict (db_metadata_cache) with:
  - db_id: stem of the sqlite filename
  - tables: list of {name, columns: [{name, type}], foreign_keys: [{from_col, to_table, to_col}],
                      sample_rows: [list[any]]}

No LLM used — pure PRAGMA queries only.
"""

import sqlite3
from pathlib import Path


def compile_schema_cache(db_path: str) -> dict:
    """Compile schema metadata from a .sqlite file via PRAGMA queries.

    Returns:
        {
          "db_id": str,
          "tables": [
            {
              "name": str,
              "columns": [{"name": str, "type": str}],
              "foreign_keys": [{"from_col": str, "to_table": str, "to_col": str}],
              "sample_rows": [list[any]]  # up to 3
            }
          ]
        }
    """
    db_id = Path(db_path).stem
    conn = sqlite3.connect(db_path)
    conn.row_factory = None
    cur = conn.cursor()

    cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    table_names = [row[0] for row in cur.fetchall()]

    tables = []
    for tname in table_names:
        # Quote identifier to handle reserved-keyword table names (e.g. "order")
        qname = '"' + tname.replace('"', '""') + '"'

        # Columns
        cur.execute(f"PRAGMA table_info({qname})")
        columns = [{"name": row[1], "type": row[2]} for row in cur.fetchall()]

        # FK edges: PRAGMA foreign_key_list returns (id, seq, table, from, to, ...)
        cur.execute(f"PRAGMA foreign_key_list({qname})")
        fk_rows = cur.fetchall()
        foreign_keys = [
            {"from_col": row[3], "to_table": row[2], "to_col": row[4]}
            for row in fk_rows
        ]

        # Sample rows — up to 3
        cur.execute(f"SELECT * FROM {qname} LIMIT 3")
        sample_rows = [list(row) for row in cur.fetchall()]

        tables.append({
            "name": tname,
            "columns": columns,
            "foreign_keys": foreign_keys,
            "sample_rows": sample_rows,
        })

    conn.close()
    return {"db_id": db_id, "tables": tables}
