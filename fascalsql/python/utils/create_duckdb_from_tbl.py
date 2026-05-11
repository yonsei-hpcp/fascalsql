#!/usr/bin/env python3
"""
Create DuckDB database from benchmark .tbl files.

Supports:
  - TPC-H: region, nation, part, supplier, partsupp, customer, orders, lineitem
  - SSB: date, customer, supplier, part, lineorder

Usage:
  python3 fascalsql/python/utils/create_duckdb_from_tbl.py \
    --benchmark tpch \
    --tbl-dir /tmp/tpch_tbl \
    --output /tmp/tpch.duckdb

  python3 fascalsql/python/utils/create_duckdb_from_tbl.py \
    --benchmark ssb \
    --tbl-dir /tmp/ssb_tbl \
    --output /tmp/ssb.duckdb
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

try:
    from fascalsql.python.data.tbl_streaming import iter_tbl_row_chunks
except ImportError:
    from tbl_streaming import iter_tbl_row_chunks  # type: ignore

try:
    import duckdb  # type: ignore
except Exception:
    duckdb = None


TPCH_TABLES: dict[str, list[tuple[str, str]]] = {
    "region": [("r_regionkey", "INTEGER"), ("r_name", "VARCHAR"), ("r_comment", "VARCHAR")],
    "nation": [
        ("n_nationkey", "INTEGER"),
        ("n_name", "VARCHAR"),
        ("n_regionkey", "INTEGER"),
        ("n_comment", "VARCHAR"),
    ],
    "part": [
        ("p_partkey", "INTEGER"),
        ("p_name", "VARCHAR"),
        ("p_mfgr", "VARCHAR"),
        ("p_brand", "VARCHAR"),
        ("p_type", "VARCHAR"),
        ("p_size", "INTEGER"),
        ("p_container", "VARCHAR"),
        ("p_retailprice", "DOUBLE"),
        ("p_comment", "VARCHAR"),
    ],
    "supplier": [
        ("s_suppkey", "INTEGER"),
        ("s_name", "VARCHAR"),
        ("s_address", "VARCHAR"),
        ("s_nationkey", "INTEGER"),
        ("s_phone", "VARCHAR"),
        ("s_acctbal", "DOUBLE"),
        ("s_comment", "VARCHAR"),
    ],
    "partsupp": [
        ("ps_partkey", "INTEGER"),
        ("ps_suppkey", "INTEGER"),
        ("ps_availqty", "INTEGER"),
        ("ps_supplycost", "DOUBLE"),
        ("ps_comment", "VARCHAR"),
    ],
    "customer": [
        ("c_custkey", "INTEGER"),
        ("c_name", "VARCHAR"),
        ("c_address", "VARCHAR"),
        ("c_nationkey", "INTEGER"),
        ("c_phone", "VARCHAR"),
        ("c_cntrycode", "INTEGER"),
        ("c_acctbal", "DOUBLE"),
        ("c_mktsegment", "VARCHAR"),
        ("c_comment", "VARCHAR"),
    ],
    "orders": [
        ("o_orderkey", "INTEGER"),
        ("o_custkey", "INTEGER"),
        ("o_orderstatus", "VARCHAR"),
        ("o_totalprice", "DOUBLE"),
        ("o_orderdate", "DATE"),
        ("o_orderpriority", "VARCHAR"),
        ("o_clerk", "VARCHAR"),
        ("o_shippriority", "INTEGER"),
        ("o_comment", "VARCHAR"),
    ],
    "lineitem": [
        ("l_orderkey", "INTEGER"),
        ("l_partkey", "INTEGER"),
        ("l_suppkey", "INTEGER"),
        ("l_linenumber", "INTEGER"),
        ("l_quantity", "DOUBLE"),
        ("l_extendedprice", "DOUBLE"),
        ("l_discount", "DOUBLE"),
        ("l_tax", "DOUBLE"),
        ("l_returnflag", "VARCHAR"),
        ("l_linestatus", "VARCHAR"),
        ("l_shipdate", "DATE"),
        ("l_commitdate", "DATE"),
        ("l_receiptdate", "DATE"),
        ("l_shipinstruct", "VARCHAR"),
        ("l_shipmode", "VARCHAR"),
        ("l_comment", "VARCHAR"),
    ],
}


SSB_TABLES: dict[str, list[tuple[str, str]]] = {
    "date": [
        ("d_datekey", "INTEGER"),
        ("d_date", "VARCHAR"),
        ("d_dayofweek", "VARCHAR"),
        ("d_month", "VARCHAR"),
        ("d_year", "INTEGER"),
        ("d_yearmonthnum", "INTEGER"),
        ("d_yearmonth", "VARCHAR"),
        ("d_daynuminweek", "INTEGER"),
        ("d_daynuminmonth", "INTEGER"),
        ("d_daynuminyear", "INTEGER"),
        ("d_monthnuminyear", "INTEGER"),
        ("d_weeknuminyear", "INTEGER"),
        ("d_sellingseason", "VARCHAR"),
        ("d_lastdayinweekfl", "INTEGER"),
        ("d_lastdayinmonthfl", "INTEGER"),
        ("d_holidayfl", "INTEGER"),
        ("d_weekdayfl", "INTEGER"),
    ],
    "customer": [
        ("c_custkey", "INTEGER"),
        ("c_name", "VARCHAR"),
        ("c_address", "VARCHAR"),
        ("c_city", "VARCHAR"),
        ("c_nation", "VARCHAR"),
        ("c_region", "VARCHAR"),
        ("c_phone", "VARCHAR"),
        ("c_mktsegment", "VARCHAR"),
    ],
    "supplier": [
        ("s_suppkey", "INTEGER"),
        ("s_name", "VARCHAR"),
        ("s_address", "VARCHAR"),
        ("s_city", "VARCHAR"),
        ("s_nation", "VARCHAR"),
        ("s_region", "VARCHAR"),
        ("s_phone", "VARCHAR"),
    ],
    "part": [
        ("p_partkey", "INTEGER"),
        ("p_name", "VARCHAR"),
        ("p_mfgr", "VARCHAR"),
        ("p_category", "VARCHAR"),
        ("p_brand1", "VARCHAR"),
        ("p_color", "VARCHAR"),
        ("p_type", "VARCHAR"),
        ("p_size", "INTEGER"),
        ("p_container", "VARCHAR"),
    ],
    "lineorder": [
        ("lo_orderkey", "INTEGER"),
        ("lo_linenumber", "INTEGER"),
        ("lo_custkey", "INTEGER"),
        ("lo_partkey", "INTEGER"),
        ("lo_suppkey", "INTEGER"),
        ("lo_orderdate", "INTEGER"),
        ("lo_orderpriority", "VARCHAR"),
        ("lo_shippriority", "INTEGER"),
        ("lo_quantity", "INTEGER"),
        ("lo_extendedprice", "INTEGER"),
        ("lo_ordtotalprice", "INTEGER"),
        ("lo_discount", "INTEGER"),
        ("lo_revenue", "INTEGER"),
        ("lo_supplycost", "INTEGER"),
        ("lo_tax", "INTEGER"),
        ("lo_commitdate", "INTEGER"),
        ("lo_shipmode", "VARCHAR"),
    ],
}


def create_table_from_tbl(
    conn: "duckdb.DuckDBPyConnection",
    table_name: str,
    columns: list[tuple[str, str]],
    tbl_path: Path,
) -> None:
    col_defs = ", ".join(f"{name} {dtype}" for name, dtype in columns)
    conn.execute(f"DROP TABLE IF EXISTS {table_name}")
    conn.execute(f"CREATE TABLE {table_name} ({col_defs})")

    names = [name for name, _ in columns]
    if table_name == "customer" and "c_cntrycode" in names:
        projection_terms = []
        for i, name in enumerate(names):
            if name == "c_cntrycode":
                projection_terms.append("CAST(substr(column5, 1, 2) AS INTEGER) AS c_cntrycode")
            else:
                src_idx = i if i > names.index("c_cntrycode") else i + 1
                projection_terms.append(f"column{src_idx} AS {name}")
        projection = ", ".join(projection_terms)
    else:
        projection = ", ".join(f"column{i + 1} AS {name}" for i, name in enumerate(names))

    columns_sql = ", ".join(
        [
            f"'column{i + 1}': 'VARCHAR'"
            for i in range((len(columns) if table_name != 'customer' or 'c_cntrycode' not in names else len(columns) - 1) + 1)
        ]
    )

    try:
        conn.execute(
            f"""
            INSERT INTO {table_name}
            SELECT {projection}
            FROM read_csv(
                '{tbl_path.as_posix()}',
                delim='|',
                header=false,
                columns={{ {columns_sql} }},
                nullstr=''
            )
            """
        )
    except Exception as exc:
        # Some benchmark generators emit occasional non-UTF-8 bytes in .tbl files.
        # Sanitize once via the shared binary-safe reader, then keep the fast DuckDB CSV path.
        if "Invalid unicode" not in str(exc):
            raise

        sanitized_path = _write_sanitized_tbl(tbl_path)
        try:
            conn.execute(
                f"""
                INSERT INTO {table_name}
                SELECT {projection}
                FROM read_csv(
                    '{sanitized_path.as_posix()}',
                    delim='|',
                    header=false,
                    columns={{ {columns_sql} }},
                    nullstr=''
                )
                """
            )
        finally:
            sanitized_path.unlink(missing_ok=True)


def _write_sanitized_tbl(tbl_path: Path) -> Path:
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, suffix=".tbl") as fout:
        sanitized_path = Path(fout.name)
        for chunk in iter_tbl_row_chunks(tbl_path, chunk_size=50_000):
            for row in chunk:
                fout.write("|".join(row))
                fout.write("|\n")
    return sanitized_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Create DuckDB database from benchmark .tbl files")
    parser.add_argument("--benchmark", choices=["tpch", "ssb"], required=True)
    parser.add_argument("--tbl-dir", required=True, help="Directory containing .tbl files")
    parser.add_argument("--output", required=True, help="Output DuckDB file path")
    args = parser.parse_args()

    if duckdb is None:
        print("Error: duckdb package is not installed.", file=sys.stderr)
        print("Install with: python3 -m pip install duckdb", file=sys.stderr)
        return 1

    tbl_dir = Path(args.tbl_dir)
    out_path = Path(args.output)

    if not tbl_dir.is_dir():
        print(f"Error: TBL directory not found: {tbl_dir}", file=sys.stderr)
        return 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()

    tables = TPCH_TABLES if args.benchmark == "tpch" else SSB_TABLES
    conn = duckdb.connect(str(out_path))

    try:
        for table_name, columns in tables.items():
            tbl_file = tbl_dir / f"{table_name}.tbl"
            if not tbl_file.exists():
                print(f"  SKIP {table_name}: {tbl_file} not found")
                continue
            print(f"  Loading {table_name} ...")
            create_table_from_tbl(conn, table_name, columns, tbl_file)

        print(f"\nCreated DuckDB: {out_path}")
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
