#!/usr/bin/env python3
"""
SSB resident FaScalSQL result verification against golden reference.

Usage:
  python3 fascalsql/python/utils/verify_ssb_results.py \\
    --data-dir /tmp/fascalsql_ssb_data \\
    --golden fascalsql/configs/verification/ssb_golden.json

Decoding:
  FaScal encodes string columns as integers during data prep (prepare_fascalsql_ssb_bin.py).
  The per-dataset encoding maps are stored as <data-dir>/<col>_dict.json (forward: str->int).
  This verifier loads those files at runtime (NOT hardcoded) so any SF / dataset works correctly.
  Column order and which dict file to use per query is specified in ssb_result_schema.json.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    from fascalsql.python.utils.verification_core import (
        build_pred_flags,
        build_query_pred_flags,
        compare_rows_with_golden,
        load_duckdb_reference_entry,
        parse_predicate_id_list,
        resolve_queries,
        run_standalone_query,
        validate_query_result_schema,
    )
except ImportError:
    from verification_core import (  # type: ignore
        build_pred_flags,
        build_query_pred_flags,
        compare_rows_with_golden,
        load_duckdb_reference_entry,
        parse_predicate_id_list,
        resolve_queries,
        run_standalone_query,
        validate_query_result_schema,
    )


# SSB queries
SSB_QUERIES = [
    "ssb_q11", "ssb_q12", "ssb_q13",
    "ssb_q21", "ssb_q22", "ssb_q23",
    "ssb_q31", "ssb_q32", "ssb_q33", "ssb_q34",
    "ssb_q41", "ssb_q42", "ssb_q43",
]


def _golden_key_candidates(query_name: str) -> List[str]:
    """Return compatible golden keys for a binary query name.

    Supports both ssb_q11 and ssb_q1_1 key styles.
    """
    candidates = [query_name]
    m = re.fullmatch(r"ssb_q([1-4])([1-4])", query_name)
    if m:
        candidates.append(f"ssb_q{m.group(1)}_{m.group(2)}")
    return candidates


def _load_reverse_dicts(data_dir: Path, schema_columns: List[dict]) -> List[Optional[Dict[str, str]]]:
    """Load reverse decode maps (int→str) for each output column from data_dir.

    data_dir contains <col>_dict.json files written by prepare_fascalsql_ssb_bin.py.
    Each dict file is a forward map {string_value: int_id}.
    We invert it here so we can decode int_id → string_value at verification time.

    Returns a list parallel to schema_columns; None for plain-int (no decode) columns.
    """
    reverse_dicts: List[Optional[Dict[str, str]]] = []
    for col_spec in schema_columns:
        dict_file = col_spec.get("dict_file")
        if dict_file:
            dict_path = data_dir / dict_file
            if dict_path.exists():
                fwd: Dict[str, int] = json.loads(dict_path.read_text())
                # invert: id (int) → string value
                rev: Dict[str, str] = {str(v): k for k, v in fwd.items()}
                reverse_dicts.append(rev)
            else:
                # dict file missing → treat as plain int (best-effort)
                reverse_dicts.append(None)
        else:
            reverse_dicts.append(None)
    return reverse_dicts


def decode_and_reorder(
    raw_rows: List[List[str]],
    schema: dict,
    data_dir: Path,
) -> List[List[str]]:
    """Decode integer-encoded columns using per-dataset dict files and reorder columns.

    schema fields:
      fascal_columns   – list of {type: "dict", dict_file: "..."} or {type: "int"}
                         in the same order FaScal prints them
      golden_column_order – list of indices into decoded fascal columns that matches
                            the golden (SQL SELECT) column order.
                            e.g. [2, 0, 1] means: golden_col0 = decoded_col2, etc.
    """
    col_specs = schema.get("fascal_columns", [])
    golden_order = schema.get("golden_column_order", list(range(len(col_specs))))

    # Load reverse dicts from the data directory (dataset-specific, not hardcoded)
    rev_dicts = _load_reverse_dicts(data_dir, col_specs)

    decoded_rows: List[List[str]] = []
    for raw_row in raw_rows:
        decoded: List[str] = []
        for i, cell in enumerate(raw_row):
            if i < len(rev_dicts) and rev_dicts[i] is not None:
                decoded.append(rev_dicts[i].get(cell.strip(), cell.strip()))
            else:
                decoded.append(cell.strip())
        # Reorder to match golden column order
        reordered = [decoded[idx] for idx in golden_order if idx < len(decoded)]
        decoded_rows.append(reordered)

    return decoded_rows


def parse_fascalsql_output(stdout: str) -> List[List[str]]:
    """Parse FaScalSQL stdout into result rows."""
    result_rows: List[List[str]] = []
    for line in stdout.splitlines():
        line = line.strip()
        if line.startswith("ROW:"):
            row = [c.strip() for c in line[4:].split("|")]
            result_rows.append(row)
        elif line.startswith("Result:"):
            result_rows.append([line.split()[-1].strip()])
    return result_rows


def _sort_key(row: List[str]):
    """Sort key that handles mixed int/string columns."""
    def cell_key(v: str):
        try:
            return (0, int(v), v)
        except ValueError:
            return (1, 0, v)
    return tuple(cell_key(v) for v in row)


def compare_results(
    fascalsql_rows: List[List[str]],
    golden_rows: List[List[str]],
    tolerance: float = 1e-6,
) -> Tuple[bool, str]:
    return compare_rows_with_golden(fascalsql_rows, {"lines": golden_rows}, tolerance=tolerance)


def run_query(
    data_dir: Path,
    query_name: str,
    pred_flags: Optional[List[int]] = None,
    timeout_s: int = 60,
) -> Tuple[Optional[List[List[str]]], Optional[str]]:
    """Run a FaScalSQL query through the resident runtime and return raw result rows or error."""
    returncode, output = run_standalone_query(
        query_name,
        pred_flags=pred_flags,
        timeout_s=timeout_s,
        data_dir=data_dir,
    )
    if returncode != 0:
        return None, f"Exit code {returncode}: {output[:200]}"
    rows = parse_fascalsql_output(output)
    if not rows:
        return None, "No output rows found"
    return rows, None


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify SSB results against golden reference")
    parser.add_argument("--data-dir", required=True, help="SSB binary data directory")
    parser.add_argument("--golden", default=None, help="Golden reference JSON")
    parser.add_argument("--duckdb-db", default=None, help="DuckDB reference DB to execute the canonical SQL against")
    parser.add_argument("--schema", default=None, help="SSB result schema JSON (column decode/order spec)")
    parser.add_argument("--tolerance", type=float, default=1e-6, help="Numeric comparison tolerance")
    parser.add_argument("--timeout", type=int, default=60, help="Per-query timeout (seconds)")
    parser.add_argument("--queries", nargs="+", default=None, help="Optional subset of queries to verify")
    parser.add_argument(
        "--pred-mode",
        choices=["gpu_all", "cpu_all", "default_cpu", "alternating", "selected_cpu"],
        default="gpu_all",
        help="Predicate placement strategy for verification runs",
    )
    parser.add_argument(
        "--cpu-predicate-ids",
        type=str,
        default=None,
        help="Comma-separated predicate ids to force onto CPU when --pred-mode=selected_cpu",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent.parent.parent
    data_dir = Path(args.data_dir)
    duckdb_db = Path(args.duckdb_db) if args.duckdb_db else None

    if not data_dir.exists():
        print(f"Error: Data directory not found: {data_dir}", file=sys.stderr)
        return 1
    if duckdb_db is not None and not duckdb_db.exists():
        print(f"Error: DuckDB reference not found: {duckdb_db}", file=sys.stderr)
        return 1

    golden: Dict[str, Dict[str, Any]] = {}
    if duckdb_db is None:
        golden_path = (
            Path(args.golden)
            if args.golden
            else repo_root / "fascalsql" / "configs" / "verification" / "ssb_golden.json"
        )
        if not golden_path.exists():
            print(f"Error: Golden reference not found: {golden_path}", file=sys.stderr)
            return 1
        print(f"Using golden:  {golden_path}")

        with open(golden_path, "r", encoding="utf-8") as f:
            golden = json.load(f)
    else:
        print(f"Using DuckDB reference: {duckdb_db}")

    # Load result schema (column decode + order spec)
    if args.schema:
        schema_path = Path(args.schema)
    else:
        schema_path = repo_root / "fascalsql" / "configs" / "verification" / "ssb_result_schema.json"
    result_schema: dict = {}
    if schema_path.exists():
        print(f"Using schema:  {schema_path}")
        result_schema = json.loads(schema_path.read_text())
    else:
        print(f"Warning: schema not found ({schema_path}), no decoding will be applied", file=sys.stderr)

    print("=" * 80)
    print("SSB Query Verification")
    print(f"  data-dir:  {data_dir}")
    print(f"  Dict files loaded dynamically from data-dir (dataset-specific encoding)")
    print("=" * 80)

    passed = 0
    failed = 0
    skipped = 0
    try:
        selected_queries = resolve_queries(args.queries, SSB_QUERIES)
        selected_cpu_ids = parse_predicate_id_list(args.cpu_predicate_ids)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    for query_name in selected_queries:
        print(f"\n{query_name}...", end=" ", flush=True)

        golden_entry: Optional[Dict[str, Any]] = None
        if duckdb_db is None:
            golden_key = None
            for cand in _golden_key_candidates(query_name):
                if cand in golden:
                    golden_key = cand
                    break
            if golden_key is None:
                print("SKIP (no golden reference)")
                skipped += 1
                continue
            golden_entry = golden[golden_key]
        else:
            try:
                golden_entry = load_duckdb_reference_entry(query_name, duckdb_db)
            except Exception as exc:
                print(f"FAIL (DuckDB reference error: {exc})")
                failed += 1
                continue

        try:
            pred_flags = build_query_pred_flags(
                query_name,
                placement_mode=args.pred_mode,
                cpu_predicate_ids=selected_cpu_ids,
            )
        except ValueError as exc:
            print(f"FAIL ({exc})")
            failed += 1
            continue

        raw_rows, error = run_query(data_dir, query_name, pred_flags=pred_flags, timeout_s=args.timeout)
        if error:
            print(f"FAIL ({error})")
            failed += 1
            continue

        # Decode + reorder using per-dataset dict files from data_dir
        query_schema = result_schema.get(query_name, {})
        if query_schema.get("fascal_columns"):
            decoded_rows = decode_and_reorder(raw_rows, query_schema, data_dir)
        else:
            decoded_rows = raw_rows

        assert golden_entry is not None
        golden_rows: List[List[str]] = golden_entry["lines"]
        schema_ok, schema_msg = validate_query_result_schema(
            benchmark="ssb",
            query=query_name,
            rows=decoded_rows,
            result_schema=result_schema,
            golden_entry=golden_entry,
        )
        if not schema_ok:
            print(f"FAIL ({schema_msg})")
            failed += 1
            continue
        match, msg = compare_results(decoded_rows, golden_rows, args.tolerance)

        if match:
            print(f"PASS ({len(decoded_rows)} rows)")
            passed += 1
        else:
            print(f"FAIL ({msg})")
            failed += 1

    print("\n" + "=" * 80)
    print(f"Summary: {passed} passed, {failed} failed, {skipped} skipped")
    print("=" * 80)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
