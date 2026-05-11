#!/usr/bin/env python3
"""
SQL Query Registry for TPC-H and SSB

Manages a registry of SQL queries for benchmark suites, parsing them from
SQL files and extracting query metadata (name, number, subquery info).

Architecture:
=============
- Parses query files (tpch_queries.sql, ssb_queries.sql)
- Extracts queries by header markers (-- Q1, -- Q1.1, etc.)
- Maps query files to query identifiers
- Validates query consistency across sources

File Format:
============
Queries are delimited by headers like:
    -- Q1 (for TPC-H)
    -- Q1.1 (for SSB subqueries)

Supported Formats:
- tpch_queries.sql: TPC-H queries Q1-Q22
- ssb_queries.sql: SSB queries Q11-Q43 (with Q1.1, Q1.2, etc. notation)

Usage Example:
==============
    from sql_query_registry import parse_query_dir
    
    queries = parse_query_dir("/path/to/queries", "tpch")
    # Returns: {"tpch_q1": "SELECT...", "tpch_q2": "SELECT...", ...}

Author: FaScalSQL Team
Last Modified: 2026-03-05
"""

import re
from pathlib import Path
from typing import Dict, List, Tuple


HEADER_RE = re.compile(r"^\s*--\s*=+\s*Q(\d+)(?:\.(\d+))?\s*[:=]?.*?=*\s*$", re.IGNORECASE)
SSB_FILE_RE = re.compile(r"^\s*(?:ssb_)?q(\d)(\d)\.sql\s*$", re.IGNORECASE)
TPCH_FILE_RE = re.compile(r"^\s*(?:tpch_)?q(\d+)\.sql\s*$", re.IGNORECASE)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _query_file_for_family(family: str) -> Path:
    root = _repo_root() / "docs" / "schemas"
    if family == "ssb":
        return root / "ssb_queries.sql"
    if family == "tpch":
        return root / "tpch_queries.sql"
    raise ValueError(f"unsupported family: {family}")


def _query_dir_for_family(family: str) -> Path:
    # Prefer fascalsql/query/sql/ (hand-written, DuckDB-compatible) over docs/schemas/
    query_sql = _repo_root() / "fascalsql" / "query" / "sql" / family
    if query_sql.exists():
        return query_sql
    doc_path = _repo_root() / "docs" / "schemas" / family
    if doc_path.exists():
        return doc_path
    raise ValueError(f"unsupported family: {family}")


def _query_name(family: str, major: str, minor: str | None) -> str:
    if family == "ssb":
        if minor is None:
            raise ValueError(f"invalid SSB query header: Q{major}")
        return f"ssb_q{major}{minor}"
    if family == "tpch":
        return f"tpch_q{int(major)}"
    raise ValueError(f"unsupported family: {family}")


def _normalize_sql_text(sql: str) -> str:
    normalized = sql.strip()
    if not normalized.endswith(";"):
        normalized = normalized + ";"
    return normalized


def parse_query_dir(path: Path, family: str) -> Dict[str, str]:
    if not path.exists() or not path.is_dir():
        return {}

    out: Dict[str, str] = {}
    for sql_path in sorted(path.iterdir()):
        if not sql_path.is_file() or sql_path.suffix.lower() != ".sql":
            continue

        query_name: str | None = None
        file_name = sql_path.name
        if family == "ssb":
            m = SSB_FILE_RE.match(file_name)
            if m:
                query_name = f"ssb_q{int(m.group(1))}{int(m.group(2))}"
        elif family == "tpch":
            m = TPCH_FILE_RE.match(file_name)
            if m:
                query_name = f"tpch_q{int(m.group(1))}"
        else:
            raise ValueError(f"unsupported family: {family}")

        if query_name is None:
            continue

        sql = _normalize_sql_text(sql_path.read_text(encoding="utf-8"))
        if not sql:
            continue
        # Stable tie-breaker when duplicate filenames map to same canonical query:
        # first lexicographic file wins.
        out.setdefault(query_name, sql)
    return out


def parse_query_file(path: Path, family: str) -> Dict[str, str]:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()

    entries: List[Tuple[str, List[str]]] = []
    current_query = ""
    current_sql: List[str] = []

    for line in lines:
        m = HEADER_RE.match(line)
        if m:
            if current_query and current_sql:
                sql_text = "\n".join(current_sql).strip()
                if sql_text:
                    entries.append((current_query, [sql_text]))
            current_query = _query_name(family, m.group(1), m.group(2))
            current_sql = []
            continue
        if not current_query:
            continue
        current_sql.append(line)

    if current_query and current_sql:
        sql_text = "\n".join(current_sql).strip()
        if sql_text:
            entries.append((current_query, [sql_text]))

    out: Dict[str, str] = {}
    for q, sql_parts in entries:
        sql = _normalize_sql_text("\n".join(sql_parts))
        out[q] = sql
    return out


def _load_family_queries(family: str) -> Dict[str, str]:
    from_dir = parse_query_dir(_query_dir_for_family(family), family)

    from_file: Dict[str, str] = {}
    query_file = _query_file_for_family(family)
    if query_file.exists():
        from_file = parse_query_file(query_file, family)

    if not from_dir:
        return from_file

    merged = dict(from_file)
    # Directory is the canonical source; file-based blocks only fill missing queries.
    merged.update(from_dir)
    return merged


def load_docs_sql_queries() -> Dict[str, str]:
    ssb = _load_family_queries("ssb")
    tpch = _load_family_queries("tpch")
    merged: Dict[str, str] = {}
    merged.update(ssb)
    merged.update(tpch)
    return merged


def sync_query_sql_files(output_root: Path) -> Dict[str, Path]:
    sql_map = load_docs_sql_queries()
    out: Dict[str, Path] = {}
    for query, sql in sorted(sql_map.items()):
        family = "ssb" if query.startswith("ssb_") else "tpch"
        dest_dir = output_root / family
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_path = dest_dir / f"{query}.sql"
        dest_path.write_text(sql + "\n", encoding="utf-8")
        out[query] = dest_path
    return out
