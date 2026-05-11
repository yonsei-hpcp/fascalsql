#!/usr/bin/env python3
"""Generate TPC-H scale-factor 1 source .tbl files using DuckDB's tpch extension.

Usage:
    python generate_tpch_sf1_tbl.py --out /tmp/tpch_sf1_tbl
    # optionally specify --sf <scale factor> (default 1)

The script creates a directory and writes all 8 TPC-H tables in pipe-delimited format
with trailing pipe on each line, matching the format expected by
`prepare_fascalsql_tpch_bin.py`.

If DuckDB or the tpch extension are not available, the script will error.
"""

import argparse
import duckdb
from pathlib import Path

TABLES = [
    "lineitem",
    "orders",
    "customer",
    "part",
    "partsupp",
    "supplier",
    "nation",
    "region",
]


def main():
    parser = argparse.ArgumentParser(description="Generate tpch .tbl files using DuckDB dbgen")
    parser.add_argument("--out", required=True, help="Output directory for generated .tbl files")
    parser.add_argument("--sf", type=float, default=1.0, help="Scale factor to generate (default 1.0)")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()
    con.execute("INSTALL tpch; LOAD tpch;")
    con.execute(f"CALL dbgen(sf={args.sf});")

    for tbl in TABLES:
        out_path = out_dir / f"{tbl}.tbl"
        print(f"Exporting {tbl} to {out_path} ...")
        # query all rows and write to file in pipe-delimited form
        rows = con.execute(f"SELECT * FROM {tbl}").fetchall()
        with open(out_path, "w", encoding="utf-8") as fout:
            for row in rows:
                fout.write("|".join(str(x) for x in row) + "|\n")
        print(f"  wrote {len(rows)} rows")

    con.close()
    print("Done")


if __name__ == "__main__":
    raise SystemExit(main())
