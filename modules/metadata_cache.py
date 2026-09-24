"""Persistent metadata mapping with JSON values; never deserialize Python code."""

import json
import os
import sqlite3
from collections.abc import MutableMapping
from contextlib import closing


def _pack(value):
    if isinstance(value, dict):
        return {"dict": [[_pack(key), _pack(item)] for key, item in value.items()]}
    if isinstance(value, tuple):
        return {"tuple": [_pack(item) for item in value]}
    if isinstance(value, list):
        return [_pack(item) for item in value]
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(f"Unsupported metadata cache value: {type(value).__name__}")


def _unpack(value):
    if isinstance(value, dict):
        if set(value) == {"dict"}:
            return {_unpack(key): _unpack(item) for key, item in value["dict"]}
        if set(value) == {"tuple"}:
            return tuple(_unpack(item) for item in value["tuple"])
        raise ValueError("Invalid metadata cache record")
    if isinstance(value, list):
        return [_unpack(item) for item in value]
    return value


class MetadataCache(MutableMapping):
    """Small process-safe mapping for regenerable hashes and model metadata.

    A separate filename deliberately leaves old pickle caches unread. They are
    not migrated or deleted: metadata is regenerated from the source files.
    Connections are short lived, so workers and application restarts share the
    cache without inheriting connections or leaking handles.
    """

    def __init__(self, directory):
        os.makedirs(directory, exist_ok=True)
        self.filename = os.path.join(directory, "metadata-v1.sqlite3")
        with closing(self._connect()) as connection, connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")

    def _connect(self):
        return sqlite3.connect(self.filename, timeout=30)

    @staticmethod
    def _key(key):
        return json.dumps(_pack(key), ensure_ascii=True, separators=(",", ":"))

    def __getitem__(self, key):
        with closing(self._connect()) as connection:
            row = connection.execute("SELECT value FROM metadata WHERE key = ?", (self._key(key),)).fetchone()
        if row is None:
            raise KeyError(key)
        return _unpack(json.loads(row[0]))

    def __setitem__(self, key, value):
        encoded = json.dumps(_pack(value), ensure_ascii=True, separators=(",", ":"))
        with closing(self._connect()) as connection, connection:
            connection.execute("INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)", (self._key(key), encoded))

    def __delitem__(self, key):
        with closing(self._connect()) as connection, connection:
            cursor = connection.execute("DELETE FROM metadata WHERE key = ?", (self._key(key),))
            if cursor.rowcount == 0:
                raise KeyError(key)

    def __iter__(self):
        with closing(self._connect()) as connection:
            rows = connection.execute("SELECT key FROM metadata ORDER BY key").fetchall()
        return (_unpack(json.loads(row[0])) for row in rows)

    def __len__(self):
        with closing(self._connect()) as connection:
            return connection.execute("SELECT count(*) FROM metadata").fetchone()[0]

    def clear(self):
        with closing(self._connect()) as connection, connection:
            connection.execute("DELETE FROM metadata")

    def close(self):
        """Compatibility: connections are already closed after each operation."""
