#!/usr/bin/env python3
"""
index_db.py — SQLite backend for storing and querying indexes.

Supports both flat chunk indexes (no edges) and graph indexes (with typed edges).
A flat index is just a degenerate graph — same schema, no rows in edges.

Schema:
    indexes — named index configurations
    files   — per-file checksums for incremental indexing
    nodes   — text chunks or symbols with optional embeddings
    edges   — typed relationships between nodes (CALLS, IMPORTS, etc.)
"""
import sqlite3
import json
import struct
from contextlib import contextmanager
from pathlib import Path

import numpy as np

DEFAULT_DB_PATH = ".maia_index.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS indexes (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    name    TEXT    NOT NULL UNIQUE,
    config  TEXT    NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS files (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    index_id     INTEGER NOT NULL REFERENCES indexes(id) ON DELETE CASCADE,
    path         TEXT    NOT NULL,
    checksum     TEXT    NOT NULL,
    last_indexed TEXT    NOT NULL DEFAULT (datetime('now')),
    UNIQUE(index_id, path)
);

CREATE TABLE IF NOT EXISTS nodes (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    index_id  INTEGER NOT NULL REFERENCES indexes(id) ON DELETE CASCADE,
    file_id   INTEGER REFERENCES files(id) ON DELETE CASCADE,
    text      TEXT    NOT NULL,
    embedding BLOB,
    metadata  TEXT    NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS edges (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    index_id  INTEGER NOT NULL REFERENCES indexes(id) ON DELETE CASCADE,
    source_id INTEGER NOT NULL REFERENCES nodes(id) ON DELETE CASCADE,
    target_id INTEGER NOT NULL REFERENCES nodes(id) ON DELETE CASCADE,
    kind      TEXT    NOT NULL,
    UNIQUE(index_id, source_id, target_id, kind)
);

CREATE INDEX IF NOT EXISTS idx_files_index    ON files(index_id);
CREATE INDEX IF NOT EXISTS idx_files_path     ON files(index_id, path);
CREATE INDEX IF NOT EXISTS idx_nodes_index    ON nodes(index_id);
CREATE INDEX IF NOT EXISTS idx_nodes_file     ON nodes(file_id);
CREATE INDEX IF NOT EXISTS idx_edges_source   ON edges(source_id);
CREATE INDEX IF NOT EXISTS idx_edges_target   ON edges(target_id);
CREATE INDEX IF NOT EXISTS idx_edges_kind     ON edges(index_id, kind);
"""


@contextmanager
def _connect(db_path: str):
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def encode_embedding(vec: np.ndarray) -> bytes:
    return struct.pack(f"{len(vec)}f", *vec)

def decode_embedding(blob: bytes) -> np.ndarray:
    n = len(blob) // 4
    return np.array(struct.unpack(f"{n}f", blob), dtype=np.float32)


def init_db(db_path: str = DEFAULT_DB_PATH):
    """Create the database and schema if they don't exist."""
    with _connect(db_path) as conn:
        conn.executescript(SCHEMA)


def create_index(name: str, config: dict, db_path: str = DEFAULT_DB_PATH) -> int:
    """
    Create a named index with the given config.
    Returns the index id.
    Raises if an index with that name already exists.
    """
    with _connect(db_path) as conn:
        cursor = conn.execute(
            "INSERT INTO indexes (name, config) VALUES (?, ?)",
            (name, json.dumps(config))
        )
        return cursor.lastrowid


def get_or_create_index(name: str, config: dict, db_path: str = DEFAULT_DB_PATH) -> int:
    """Get an existing index by name or create it. Returns the index id."""
    with _connect(db_path) as conn:
        row = conn.execute(
            "SELECT id FROM indexes WHERE name = ?", (name,)
        ).fetchone()
        if row:
            return row["id"]
        cursor = conn.execute(
            "INSERT INTO indexes (name, config) VALUES (?, ?)",
            (name, json.dumps(config))
        )
        return cursor.lastrowid


def list_indexes(db_path: str = DEFAULT_DB_PATH) -> list[dict]:
    """List all indexes with their configs and node/edge counts."""
    with _connect(db_path) as conn:
        rows = conn.execute("""
            SELECT
                i.id, i.name, i.config,
                COUNT(DISTINCT n.id) AS node_count,
                COUNT(DISTINCT e.id) AS edge_count,
                COUNT(DISTINCT f.id) AS file_count
            FROM indexes i
            LEFT JOIN nodes n ON n.index_id = i.id
            LEFT JOIN edges e ON e.index_id = i.id
            LEFT JOIN files f ON f.index_id = i.id
            GROUP BY i.id
        """).fetchall()
        return [dict(r) for r in rows]


def delete_index(name: str, db_path: str = DEFAULT_DB_PATH):
    """Delete a named index and all its files, nodes, and edges."""
    with _connect(db_path) as conn:
        conn.execute("DELETE FROM indexes WHERE name = ?", (name,))


def get_file_checksum(index_id: int, path: str, db_path: str = DEFAULT_DB_PATH) -> str | None:
    """Return the stored checksum for a file in an index, or None if not indexed."""
    with _connect(db_path) as conn:
        row = conn.execute(
            "SELECT checksum FROM files WHERE index_id = ? AND path = ?",
            (index_id, path)
        ).fetchone()
        return row["checksum"] if row else None


def upsert_file(index_id: int, path: str, checksum: str, db_path: str = DEFAULT_DB_PATH) -> int:
    """
    Insert or replace a file record.
    If replacing, cascades to delete all nodes (and their edges) for this file.
    Returns the file id.
    """
    with _connect(db_path) as conn:
        # delete existing to trigger cascade on nodes/edges
        conn.execute(
            "DELETE FROM files WHERE index_id = ? AND path = ?",
            (index_id, path)
        )
        cursor = conn.execute(
            "INSERT INTO files (index_id, path, checksum) VALUES (?, ?, ?)",
            (index_id, path, checksum)
        )
        return cursor.lastrowid


def insert_node(
    index_id: int,
    text: str,
    metadata: dict,
    file_id: int | None = None,
    embedding: np.ndarray | None = None,
    db_path: str = DEFAULT_DB_PATH,
) -> int:
    """Insert a node and return its id."""
    blob = encode_embedding(embedding) if embedding is not None else None
    with _connect(db_path) as conn:
        cursor = conn.execute(
            "INSERT INTO nodes (index_id, file_id, text, embedding, metadata) VALUES (?, ?, ?, ?, ?)",
            (index_id, file_id, text, blob, json.dumps(metadata))
        )
        return cursor.lastrowid


def insert_nodes(
    index_id: int,
    nodes: list[dict],
    db_path: str = DEFAULT_DB_PATH,
) -> list[int]:
    """
    Bulk insert nodes. Each node dict should have:
        text, metadata, file_id (optional), embedding (optional np.ndarray)
    Returns list of inserted ids.
    """
    with _connect(db_path) as conn:
        ids = []
        for node in nodes:
            blob = encode_embedding(node["embedding"]) if node.get("embedding") is not None else None
            cursor = conn.execute(
                "INSERT INTO nodes (index_id, file_id, text, embedding, metadata) VALUES (?, ?, ?, ?, ?)",
                (
                    index_id,
                    node.get("file_id"),
                    node["text"],
                    blob,
                    json.dumps(node.get("metadata", {})),
                )
            )
            ids.append(cursor.lastrowid)
        return ids


def update_node_embedding(node_id: int, embedding: np.ndarray, db_path: str = DEFAULT_DB_PATH):
    """Store or update the embedding for a node."""
    with _connect(db_path) as conn:
        conn.execute(
            "UPDATE nodes SET embedding = ? WHERE id = ?",
            (encode_embedding(embedding), node_id)
        )


def get_nodes_without_embeddings(index_id: int, db_path: str = DEFAULT_DB_PATH) -> list[dict]:
    """Return all nodes in an index that don't have embeddings yet."""
    with _connect(db_path) as conn:
        rows = conn.execute(
            "SELECT id, text, metadata FROM nodes WHERE index_id = ? AND embedding IS NULL",
            (index_id,)
        ).fetchall()
        return [dict(r) for r in rows]


def get_all_nodes_with_embeddings(index_id: int, db_path: str = DEFAULT_DB_PATH) -> list[dict]:
    """Return all nodes in an index that have embeddings, with decoded vectors."""
    with _connect(db_path) as conn:
        rows = conn.execute(
            "SELECT id, text, embedding, metadata FROM nodes WHERE index_id = ? AND embedding IS NOT NULL",
            (index_id,)
        ).fetchall()
        return [
            {
                "id": r["id"],
                "text": r["text"],
                "embedding": decode_embedding(r["embedding"]),
                "metadata": json.loads(r["metadata"]),
            }
            for r in rows
        ]


def insert_edge(
    index_id: int,
    source_id: int,
    target_id: int,
    kind: str,
    db_path: str = DEFAULT_DB_PATH,
):
    """Insert an edge, ignoring duplicates."""
    with _connect(db_path) as conn:
        conn.execute(
            "INSERT OR IGNORE INTO edges (index_id, source_id, target_id, kind) VALUES (?, ?, ?, ?)",
            (index_id, source_id, target_id, kind)
        )


def insert_edges(index_id: int, edges: list[dict], db_path: str = DEFAULT_DB_PATH):
    """
    Bulk insert edges. Each edge dict should have:
        source_id, target_id, kind
    """
    with _connect(db_path) as conn:
        conn.executemany(
            "INSERT OR IGNORE INTO edges (index_id, source_id, target_id, kind) VALUES (?, ?, ?, ?)",
            [(index_id, e["source_id"], e["target_id"], e["kind"]) for e in edges]
        )


def get_neighbors(
    node_id: int,
    index_id: int,
    kinds: list[str] | None = None,
    direction: str = "out",
    db_path: str = DEFAULT_DB_PATH,
) -> list[dict]:
    """
    Get neighboring nodes via edges.

    direction: 'out' (source=node_id), 'in' (target=node_id), 'both'
    kinds: filter to specific edge types, None = all
    """
    params = [index_id]
    kind_filter = ""

    if kinds:
        placeholders = ",".join("?" * len(kinds))
        kind_filter = f" AND kind IN ({placeholders})"

    if direction == "out":
        q = f"""
            SELECT n.id, n.text, n.metadata, e.kind
            FROM edges e JOIN nodes n ON n.id = e.target_id
            WHERE e.index_id = ? AND e.source_id = ?{kind_filter}
        """
        params += [node_id] + (kinds or [])
    elif direction == "in":
        q = f"""
            SELECT n.id, n.text, n.metadata, e.kind
            FROM edges e JOIN nodes n ON n.id = e.source_id
            WHERE e.index_id = ? AND e.target_id = ?{kind_filter}
        """
        params += [node_id] + (kinds or [])
    else:  # both
        q = f"""
            SELECT n.id, n.text, n.metadata, e.kind
            FROM edges e JOIN nodes n ON n.id = e.target_id
            WHERE e.index_id = ? AND e.source_id = ?{kind_filter}
            UNION
            SELECT n.id, n.text, n.metadata, e.kind
            FROM edges e JOIN nodes n ON n.id = e.source_id
            WHERE e.index_id = ? AND e.target_id = ?{kind_filter}
        """
        params = [index_id, node_id] + (kinds or []) + [index_id, node_id] + (kinds or [])

    with _connect(db_path) as conn:
        rows = conn.execute(q, params).fetchall()
        return [
            {
                "id": r["id"],
                "text": r["text"],
                "metadata": json.loads(r["metadata"]),
                "edge_kind": r["kind"],
            }
            for r in rows
        ]


def stats(db_path: str = DEFAULT_DB_PATH) -> dict:
    """Return counts per index for observability."""
    with _connect(db_path) as conn:
        rows = conn.execute("""
            SELECT
                i.name,
                COUNT(DISTINCT f.id) AS files,
                COUNT(DISTINCT n.id) AS nodes,
                COUNT(DISTINCT e.id) AS edges,
                SUM(CASE WHEN n.embedding IS NOT NULL THEN 1 ELSE 0 END) AS embedded
            FROM indexes i
            LEFT JOIN files f ON f.index_id = i.id
            LEFT JOIN nodes n ON n.index_id = i.id
            LEFT JOIN edges e ON e.index_id = i.id
            GROUP BY i.id
        """).fetchall()
        return {r["name"]: dict(r) for r in rows}
