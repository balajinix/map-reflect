"""
SQLite3 version override for systems with sqlite3 < 3.35.0.

ChromaDB requires sqlite3 >= 3.35.0. On older systems (e.g., RHEL 8 with
Python 3.9), the system sqlite3 is too old. This module replaces the
sqlite3 module with pysqlite3-binary which bundles a newer sqlite3.

Import this module before any chromadb import:
    import match_maker.sqlite_override  # noqa: F401
"""
import sys

try:
    __import__("pysqlite3")
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass  # pysqlite3 not available; hope system sqlite3 is new enough
