#!/usr/bin/env python3
"""
Database Schema Catalog for TPC-H and SSB

Parses and manages table schemas (columns, primary keys, foreign keys)
from DDL files. Used for query validation and code generation.

Architecture:
=============
- TableSchema: Represents a single table's schema
- SchemaCatalog: Repository of all table schemas
- Parsing: Extracts schemas from CREATE TABLE statements

Data Source:
============
Schema DDL files are stored in:
- docs/schemas/tpch.ddl (TPC-H)
- docs/schemas/ssb.ddl (SSB)

Features:
=========
- Parse CREATE TABLE statements
- Extract column names and types
- Identify primary and foreign keys
- Query column type information
- Validate table/column existence

Usage Example:
==============
    from schema_catalog import load_schema_catalog
    
    # Load schemas for a benchmark
    catalog = load_schema_catalog("tpch")
    
    # Get table schema
    schema = catalog.get_table("lineitem")
    
    # Check column type
    col_type = schema.columns.get("l_quantity")  # Returns: "int"
    
    # Get primary keys
    pkeys = schema.primary_keys  # Returns: {"l_orderkey", "l_linenumber"}

Author: FaScalSQL Team
Last Modified: 2026-03-05
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Set, Tuple


CREATE_TABLE_RE = re.compile(
    r"CREATE\s+TABLE\s+([A-Za-z_][A-Za-z0-9_]*)\s*\((.*?)\)\s*;",
    re.IGNORECASE | re.DOTALL,
)
COLUMN_RE = re.compile(
    r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s+([A-Za-z]+(?:\([^)]*\))?)",
    re.IGNORECASE,
)
PK_RE = re.compile(r"PRIMARY\s+KEY\s*\(([^)]+)\)", re.IGNORECASE)
FK_RE = re.compile(
    r"FOREIGN\s+KEY\s*\(([^)]+)\)\s+REFERENCES\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(([^)]+)\)",
    re.IGNORECASE,
)


@dataclass
class TableSchema:
    name: str
    columns: Dict[str, str] = field(default_factory=dict)
    primary_keys: Set[str] = field(default_factory=set)


@dataclass
class SchemaCatalog:
    tables: Dict[str, TableSchema]
    foreign_keys: Dict[Tuple[str, str], Tuple[str, str]]
    column_to_tables: Dict[str, Set[str]]

    def resolve_column(self, column: str, candidate_tables: Optional[Set[str]] = None) -> Optional[str]:
        col = str(column).strip()
        candidates = set(self.column_to_tables.get(col, set()))
        if not candidates and col.endswith("_enc"):
            candidates = set(self.column_to_tables.get(col[:-4], set()))
        if candidate_tables is not None:
            candidates &= set(candidate_tables)
        if len(candidates) == 1:
            return next(iter(candidates))
        return None

    def column_type(self, table: str, column: str) -> Optional[str]:
        t = str(table).strip()
        c = str(column).strip()
        ts = self.tables.get(t)
        if ts and c in ts.columns:
            return ts.columns[c]
        if ts and c.endswith("_enc") and c[:-4] in ts.columns:
            return ts.columns[c[:-4]]
        return None

    def is_fk_pair(self, src_table: str, src_col: str, dst_table: str, dst_col: str) -> bool:
        return self.foreign_keys.get((src_table, src_col)) == (dst_table, dst_col)

    def is_primary_key(self, table: str, column: str) -> bool:
        ts = self.tables.get(str(table).strip())
        if not ts:
            return False
        return str(column).strip() in ts.primary_keys


def _split_top_level_items(block: str) -> list[str]:
    items: list[str] = []
    cur: list[str] = []
    depth = 0
    for ch in block:
        if ch == "(":
            depth += 1
        elif ch == ")" and depth > 0:
            depth -= 1
        if ch == "," and depth == 0:
            item = "".join(cur).strip()
            if item:
                items.append(item)
            cur = []
        else:
            cur.append(ch)
    tail = "".join(cur).strip()
    if tail:
        items.append(tail)
    return items


def _strip_sql_comments(sql_text: str) -> str:
    lines = []
    for line in sql_text.splitlines():
        if "--" in line:
            line = line.split("--", 1)[0]
        lines.append(line)
    return "\n".join(lines)


def _parse_ddl_text(sql_text: str) -> tuple[Dict[str, TableSchema], Dict[Tuple[str, str], Tuple[str, str]]]:
    tables: Dict[str, TableSchema] = {}
    foreign_keys: Dict[Tuple[str, str], Tuple[str, str]] = {}

    cleaned = _strip_sql_comments(sql_text)
    for match in CREATE_TABLE_RE.finditer(cleaned):
        table_name = match.group(1).strip().lower()
        body = match.group(2)
        ts = tables.setdefault(table_name, TableSchema(name=table_name))

        for raw_item in _split_top_level_items(body):
            item = raw_item.strip()
            if not item:
                continue
            upper = item.upper()

            fk_match = FK_RE.search(item)
            if fk_match:
                src_cols = [c.strip().lower() for c in fk_match.group(1).split(",")]
                ref_table = fk_match.group(2).strip().lower()
                dst_cols = [c.strip().lower() for c in fk_match.group(3).split(",")]
                for src_col, dst_col in zip(src_cols, dst_cols):
                    foreign_keys[(table_name, src_col)] = (ref_table, dst_col)
                continue

            pk_match = PK_RE.search(item)
            if pk_match:
                for pk_col in pk_match.group(1).split(","):
                    ts.primary_keys.add(pk_col.strip().lower())
                continue

            if upper.startswith("CONSTRAINT "):
                continue

            col_match = COLUMN_RE.match(item)
            if not col_match:
                continue
            col_name = col_match.group(1).strip().lower()
            col_type = col_match.group(2).strip().upper()
            ts.columns.setdefault(col_name, col_type)

    return tables, foreign_keys


def _merge_catalogs(
    catalogs: list[tuple[Dict[str, TableSchema], Dict[Tuple[str, str], Tuple[str, str]]]]
) -> SchemaCatalog:
    tables: Dict[str, TableSchema] = {}
    foreign_keys: Dict[Tuple[str, str], Tuple[str, str]] = {}

    for tmap, fkmap in catalogs:
        for tname, ts in tmap.items():
            if tname not in tables:
                tables[tname] = TableSchema(name=tname)
            target = tables[tname]
            target.columns.update(ts.columns)
            target.primary_keys |= set(ts.primary_keys)
        foreign_keys.update(fkmap)

    column_to_tables: Dict[str, Set[str]] = {}
    for tname, ts in tables.items():
        for col in ts.columns:
            column_to_tables.setdefault(col, set()).add(tname)

    return SchemaCatalog(tables=tables, foreign_keys=foreign_keys, column_to_tables=column_to_tables)


_SCHEMA_CATALOG: Optional[SchemaCatalog] = None


def build_schema_catalog_from_ddl_text(sql_text: str) -> SchemaCatalog:
    tables, foreign_keys = _parse_ddl_text(sql_text)
    return _merge_catalogs([(tables, foreign_keys)])


def build_schema_catalog_from_paths(paths: list[Path]) -> SchemaCatalog:
    catalogs: list[tuple[Dict[str, TableSchema], Dict[Tuple[str, str], Tuple[str, str]]]] = []
    for ddl_path in paths:
        if ddl_path.exists():
            catalogs.append(_parse_ddl_text(ddl_path.read_text(encoding="utf-8")))
    if not catalogs:
        raise RuntimeError("No schema DDL files found for requested paths")
    return _merge_catalogs(catalogs)


def load_schema_catalog() -> SchemaCatalog:
    global _SCHEMA_CATALOG
    if _SCHEMA_CATALOG is not None:
        return _SCHEMA_CATALOG

    repo_root = Path(__file__).resolve().parents[3]
    schema_dir = repo_root / "docs" / "schemas"
    ddl_files = [
        schema_dir / "ssb.ddl",
        schema_dir / "tpch.ddl",
    ]
    catalogs = []
    for ddl_file in ddl_files:
        if ddl_file.exists():
            catalogs.append(_parse_ddl_text(ddl_file.read_text(encoding="utf-8")))
    if not catalogs:
        raise RuntimeError(f"No schema DDL files found under {schema_dir}")
    _SCHEMA_CATALOG = _merge_catalogs(catalogs)
    return _SCHEMA_CATALOG
