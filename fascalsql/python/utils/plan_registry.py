#!/usr/bin/env python3
"""
Query Plan Registry

Manages the central registry of query metadata for SQL-driven code generation.

Architecture:
=============
- Registry Snapshot: built directly from fascalsql/query/sql/**/*.sql
- Optional Export: fascalsql/query/plans/plan_registry.json
- Metadata: Query names, SQL paths, source, status

Registry Format:
================
{
    "version": 1,
    "entries": [
        {
            "query": "tpch_q1",
            "family": "tpch",
            "sql_path": "fascalsql/query/sql/tpch/tpch_q1.sql",
            "status": "ready"
        }
    ]
}

Usage Example:
==============
    from plan_registry import build_registry, get_query_ir
    
    # Load registry
    registry = build_registry()
    
    # Get query IR plan
    ir_plan = get_query_ir("tpch_q6")

Author: FaScalSQL Team
Last Modified: 2026-03-05
"""

import json
from pathlib import Path
from typing import Optional

try:
    from sql_query_registry import sync_query_sql_files
except ImportError:
    from fascalsql.python.utils.sql_query_registry import sync_query_sql_files  # type: ignore


REPO_ROOT = Path(__file__).resolve().parents[3]
REGISTRY_PATH = REPO_ROOT / "fascalsql" / "query" / "plans" / "plan_registry.json"
SQL_ROOT = REPO_ROOT / "fascalsql" / "query" / "sql"

SSB_QUERIES = [
    "ssb_q11",
    "ssb_q12",
    "ssb_q13",
    "ssb_q21",
    "ssb_q22",
    "ssb_q23",
    "ssb_q31",
    "ssb_q32",
    "ssb_q33",
    "ssb_q34",
    "ssb_q41",
    "ssb_q42",
    "ssb_q43",
]


def _query_family(query_name: str) -> str:
    if query_name.startswith("ssb_"):
        return "ssb"
    if query_name.startswith("tpch_"):
        return "tpch"
    return "unknown"


def _default_pipeline() -> list[str]:
    return ["scan", "selection", "join", "aggregation", "materialize"]


def _scan_sql_root() -> list[dict]:
    entries: list[dict] = []
    for family in ("tpch", "ssb"):
        family_root = SQL_ROOT / family
        if not family_root.exists():
            continue
        for sql_path in sorted(family_root.glob("*.sql")):
            query_name = sql_path.stem
            entries.append(
                {
                    "query": query_name,
                    "family": family,
                    "source": "sql",
                    "plan_path": None,
                    "sql_path": str(sql_path.relative_to(REPO_ROOT)),
                    "pipeline": _default_pipeline(),
                    "status": "ready",
                }
            )
    return entries


def build_registry() -> dict:
    entries = _scan_sql_root()
    registry = {
        "version": 1,
        "entries": sorted(entries, key=lambda x: x["query"]),
    }
    return registry


def save_registry(registry: dict) -> Path:
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    REGISTRY_PATH.write_text(json.dumps(registry, indent=2))
    return REGISTRY_PATH


def load_registry(*, persist_if_missing: bool = False) -> dict:
    if REGISTRY_PATH.exists():
        return json.loads(REGISTRY_PATH.read_text())
    registry = build_registry()
    if persist_if_missing:
        save_registry(registry)
    return registry


def get_entry(query_name: str) -> Optional[dict]:
    registry = load_registry()
    for entry in registry["entries"]:
        if entry["query"] == query_name:
            return entry
    return None


if __name__ == "__main__":
    out = save_registry(build_registry())
    print(f"Wrote plan registry: {out}")
