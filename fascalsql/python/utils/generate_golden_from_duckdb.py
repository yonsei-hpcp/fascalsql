#!/usr/bin/env python3
"""
Generate golden JSON from a DuckDB reference database.

Supports both TPC-H and SSB query sets:
  - benchmark=tpch -> tpch_q1 ... tpch_q22
  - benchmark=ssb  -> ssb_q1_1 ... ssb_q4_3

Usage:
  python3 fascalsql/python/utils/generate_golden_from_duckdb.py \
    --benchmark tpch \
    --duckdb-db /tmp/tpch.duckdb \
    --output fascalsql/configs/verification/tpch_golden.json

  python3 fascalsql/python/utils/generate_golden_from_duckdb.py \
    --benchmark ssb \
    --duckdb-db /tmp/ssb.duckdb \
    --output fascalsql/configs/verification/ssb_golden.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

try:
    import duckdb  # type: ignore
except Exception:
    duckdb = None


_IDENT_RE = re.compile(r"^[A-Za-z_]\w*$")


def fmt(val: object) -> str:
    if val is None:
        return ""
    if isinstance(val, float):
        return f"{val:.6f}"
    return str(val)


def _split_top_level(block: str) -> list[str]:
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


def _find_top_level_keyword(sql: str, keyword: str, start: int = 0) -> int:
    upper = sql.upper()
    target = keyword.upper()
    depth = 0
    in_single = False
    in_double = False
    i = start
    while i < len(sql):
        ch = sql[i]
        if ch == "'" and not in_double:
            if in_single and i + 1 < len(sql) and sql[i + 1] == "'":
                i += 2
                continue
            in_single = not in_single
        elif ch == '"' and not in_single:
            in_double = not in_double
        elif not in_single and not in_double:
            if ch == "(":
                depth += 1
            elif ch == ")" and depth > 0:
                depth -= 1
            elif depth == 0 and upper.startswith(target, i):
                before = upper[i - 1] if i > 0 else " "
                after_pos = i + len(target)
                after = upper[after_pos] if after_pos < len(upper) else " "
                if not (before.isalnum() or before == "_") and not (after.isalnum() or after == "_"):
                    return i
        i += 1
    return -1


def _expand_group_by_aliases_for_duckdb(sql: str) -> str:
    select_pos = _find_top_level_keyword(sql, "SELECT")
    if select_pos < 0:
        return sql
    from_pos = _find_top_level_keyword(sql, "FROM", select_pos + len("SELECT"))
    if from_pos < 0:
        return sql
    alias_map: dict[str, str] = {}
    for item in _split_top_level(sql[select_pos + len("SELECT"):from_pos]):
        match = re.match(r"(?is)(.+?)\s+AS\s+([A-Za-z_]\w*)\s*$", item.strip())
        if match:
            alias_map[match.group(2).lower()] = match.group(1).strip()
    if not alias_map:
        return sql
    group_pos = _find_top_level_keyword(sql, "GROUP BY", from_pos + len("FROM"))
    if group_pos < 0:
        return sql
    clause_start = group_pos + len("GROUP BY")
    clause_end = len(sql)
    for stop_keyword in ("HAVING", "ORDER BY", "LIMIT", "UNION", "EXCEPT", "INTERSECT", "QUALIFY"):
        stop_pos = _find_top_level_keyword(sql, stop_keyword, clause_start)
        if stop_pos >= 0:
            clause_end = min(clause_end, stop_pos)
    changed = False
    rewritten_items: list[str] = []
    for item in _split_top_level(sql[clause_start:clause_end]):
        token = item.strip()
        replacement = alias_map.get(token.lower()) if _IDENT_RE.match(token) else None
        if replacement:
            rewritten_items.append(replacement)
            changed = True
        else:
            rewritten_items.append(token)
    if not changed:
        return sql
    suffix = sql[clause_end:]
    separator = "" if not suffix or suffix[:1].isspace() else " "
    return f"{sql[:clause_start]} {', '.join(rewritten_items)}{separator}{suffix}"


def normalize_sql_for_duckdb(sql: str) -> str:
    """Translate SQLite-style literals/functions to DuckDB-compatible SQL."""
    def _replace_date_with_modifier(match: re.Match[str]) -> str:
        base = match.group(1)
        sign = match.group(2)
        amount = int(match.group(3))
        unit = match.group(4).upper()
        if sign == "-":
            amount = -amount
        unit = unit.rstrip("S")
        return f"(DATE '{base}' + INTERVAL {amount} {unit})"

    # SQLite: date('1994-01-01', '+1 year') -> DuckDB: (DATE '1994-01-01' + INTERVAL 1 YEAR)
    sql = re.sub(
        r"(?i)\bdate\s*\(\s*'([^']+)'\s*,\s*'([+-])\s*(\d+)\s*(day|days|month|months|year|years)'\s*\)",
        _replace_date_with_modifier,
        sql,
    )

    # SQLite: date('1998-09-02') -> DuckDB: DATE '1998-09-02'
    sql = re.sub(
        r"(?i)\bdate\s*\(\s*'([^']+)'\s*\)",
        lambda m: f"DATE '{m.group(1)}'",
        sql,
    )
    # Remove ::fixeddecimal casts (not supported by DuckDB)
    sql = re.sub(r"'(\d+(?:\.\d+)?)'::fixeddecimal", r"\1", sql)
    sql = re.sub(r"(\d+(?:\.\d+)?)::fixeddecimal", r"\1", sql)
    sql = _expand_group_by_aliases_for_duckdb(sql)
    return sql


def run_query(conn: "duckdb.DuckDBPyConnection", sql: str) -> tuple[list[list[str]], list[dict[str, str]]]:
    cursor = conn.execute(normalize_sql_for_duckdb(sql))
    rows = cursor.fetchall()
    schema = []
    for desc in cursor.description or []:
        name = str(desc[0]) if desc else ""
        duckdb_type = str(desc[1]) if len(desc) > 1 else ""
        schema.append({"name": name, "duckdb_type": duckdb_type})
    return [[fmt(cell) for cell in row] for row in rows], schema


def iter_tpch_queries(sql_dir: Path):
    for i in range(1, 23):
        query_name = f"tpch_q{i}"
        sql_file = sql_dir / f"q{i}.sql"
        if not sql_file.exists():
            sql_file = sql_dir / f"tpch_q{i}.sql"
        yield query_name, sql_file


def iter_ssb_queries(sql_dir: Path):
    for flight in range(1, 5):
        for variant in range(1, 5):
            if flight != 3 and variant > 3:
                continue
            query_name = f"ssb_q{flight}{variant}"
            sql_file = sql_dir / f"{query_name}.sql"
            yield query_name, sql_file


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate golden JSON from DuckDB reference DB.")
    parser.add_argument("--benchmark", choices=["tpch", "ssb"], required=True)
    parser.add_argument("--duckdb-db", required=True, help="Path to DuckDB database file")
    parser.add_argument(
        "--sql-dir",
        default=None,
        help="Directory containing query SQL files (default: docs/schemas/{benchmark}/)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSON path (default: fascalsql/configs/verification/{tpch,ssb}_golden.json)",
    )
    args = parser.parse_args()

    if duckdb is None:
        print("Error: duckdb package is not installed.", file=sys.stderr)
        print("Install with: pip install duckdb", file=sys.stderr)
        return 1

    db_path = Path(args.duckdb_db)
    if not db_path.exists():
        print(f"Error: DuckDB file not found: {db_path}", file=sys.stderr)
        return 1

    repo_root = Path(__file__).resolve().parent.parent.parent.parent
    sql_dir = Path(args.sql_dir) if args.sql_dir else repo_root / "fascalsql" / "query" / "sql" / args.benchmark
    if args.output:
        output_path = Path(args.output)
    elif args.benchmark == "ssb":
        output_path = repo_root / "fascalsql" / "configs" / "verification" / "ssb_golden.json"
    else:
        output_path = repo_root / "fascalsql" / "configs" / "verification" / "tpch_golden.json"

    if not sql_dir.is_dir():
        print(f"Error: SQL directory not found: {sql_dir}", file=sys.stderr)
        return 1

    conn = duckdb.connect(str(db_path), read_only=True)
    golden: dict[str, dict] = {}
    failed: list[str] = []

    query_iter = iter_tpch_queries(sql_dir) if args.benchmark == "tpch" else iter_ssb_queries(sql_dir)

    for query_name, sql_file in query_iter:
        if not sql_file.exists():
            print(f"  SKIP {query_name}: {sql_file} not found")
            continue
        sql = sql_file.read_text(encoding="utf-8")
        try:
            rows, schema = run_query(conn, sql)
            golden[query_name] = {"lines": rows, "schema": schema}
            print(f"  OK   {query_name}: {len(rows)} rows")
        except Exception as e:
            print(f"  FAIL {query_name}: {e}", file=sys.stderr)
            failed.append(query_name)

    conn.close()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(golden, f, indent=2, ensure_ascii=False)

    print(f"\nSaved {len(golden)} queries to {output_path}")
    if failed:
        print(f"Failed: {', '.join(failed)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
