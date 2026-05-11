#!/usr/bin/env python3
"""
Prepare FaScalSQL SSB .bin column files from .tbl source data.

Fully DDL-driven: reads docs/schemas/ssb.ddl to determine column names and order.
No external policy files required.

Encoding is auto-detected from data:
  - All values parse as integer  → store as raw int (predicate file)
                                    + 0-based rank (enc file, for GROUP BY)
  - "MonYYYY" pattern (d_yearmonth) → YYYYMM integer (e.g. Jan1992 → 199201)
  - Otherwise                    → dict encode (rank in alphabetical order)

Outputs per dimension column:
  - {col}.bin       : filter-predicate values (raw int for ints, dict-rank for strings)
  - {col}_enc.bin   : GROUP BY index (0-based rank)
  - {col}_dict.json : forward map {string → int} (string columns only)

Fact table (lineorder):
  - {col}.bin : raw integer values for all INTEGER columns

Also writes:
  - ssb_decode_meta.json : column-type metadata and reverse maps
"""

import argparse
import json
import re
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

try:
    from fascalsql.python.data.tbl_streaming import iter_tbl_rows, parse_tbl_line
except ImportError:
    from tbl_streaming import iter_tbl_rows, parse_tbl_line  # type: ignore

# Generic cardinality threshold: dict columns with more distinct values
# than this do NOT get _enc.bin (too large for hash-table GROUP BY).
# This is a data-only heuristic, not driven by any specific query.
_MAX_ENC_DISTINCT = 1000


# ---------------------------------------------------------------------------
# DDL parsing
# ---------------------------------------------------------------------------

_CREATE_TABLE_RE = re.compile(
    r"CREATE\s+TABLE\s+([A-Za-z_]\w*)\s*\((.*?)\)\s*;",
    re.IGNORECASE | re.DOTALL,
)
_COLUMN_RE = re.compile(r"^\s*([A-Za-z_]\w*)\s+([A-Za-z]+(?:\([^)]*\))?)", re.IGNORECASE)
_CONSTRAINT_KEYWORDS = {"primary", "foreign", "constraint", "unique", "check"}


def _strip_comments(text: str) -> str:
    return "\n".join(
        line.split("--", 1)[0] for line in text.splitlines()
    )


def _split_top_level(block: str) -> List[str]:
    items, cur, depth = [], [], 0
    for ch in block:
        if ch == "(":
            depth += 1
        elif ch == ")" and depth > 0:
            depth -= 1
        if ch == "," and depth == 0:
            s = "".join(cur).strip()
            if s:
                items.append(s)
            cur = []
        else:
            cur.append(ch)
    tail = "".join(cur).strip()
    if tail:
        items.append(tail)
    return items


def _parse_ddl(ddl_text: str) -> Dict[str, List[str]]:
    """Return {table → [col_name, ...]} preserving DDL column order."""
    tables: Dict[str, List[str]] = {}
    for m in _CREATE_TABLE_RE.finditer(_strip_comments(ddl_text)):
        tname = m.group(1).strip().lower()
        cols: List[str] = []
        for item in _split_top_level(m.group(2)):
            item = item.strip()
            if not item:
                continue
            first = item.split()[0].lower() if item.split() else ""
            if first in _CONSTRAINT_KEYWORDS:
                continue
            cm = _COLUMN_RE.match(item)
            if cm:
                cols.append(cm.group(1).strip().lower())
        tables[tname] = cols
    return tables


# ---------------------------------------------------------------------------
# .tbl parsing helpers
# ---------------------------------------------------------------------------

_MONTH_ABBR = {
    "JAN": 1, "FEB": 2, "MAR": 3,  "APR": 4,
    "MAY": 5, "JUN": 6, "JUL": 7,  "AUG": 8,
    "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
}
_YEARMONTH_RE = re.compile(r"^([A-Za-z]{3})(\d{4})$")
_STRING_ENCODING_KINDS = frozenset({"dict", "prefix_hash_int", "prefix_hash_padded_int"})


def _parse_tbl_line(line: str) -> List[str]:
    return parse_tbl_line(line)


def _norm(t: str) -> str:
    return t.strip().strip('"')


def _try_int(t: str) -> Optional[int]:
    t = _norm(t)
    if not t:
        return 0
    # Strip spaces inside (e.g. "1 000" → unlikely but safe)
    if t.lstrip("-").replace(" ", "").isdigit():
        return int(t.replace(" ", ""))
    try:
        return int(t)
    except ValueError:
        return None


def _try_yearmonth(t: str) -> Optional[int]:
    m = _YEARMONTH_RE.fullmatch(_norm(t))
    if not m:
        return None
    month = _MONTH_ABBR.get(m.group(1).upper())
    if month is None:
        return None
    return int(m.group(2)) * 100 + month


# ---------------------------------------------------------------------------
# PREFIX#NUMBER compact encoding detection
# ---------------------------------------------------------------------------

_PREFIX_HASH_INT_RE = re.compile(r'^(.+#)(\d+)$')


def _detect_prefix_hash_int(
    unique_sorted_strs: List[str],
) -> Optional[Tuple[str, List[int]]]:
    """Return (prefix, nums) if all values match PREFIX#INTEGER pattern.

    nums is sorted by str() representation of the integer suffix, which
    matches the alphabetical sort order of the full strings (same prefix).
    Only columns used for GROUP BY (not arbitrary string columns) should
    produce this compact encoding – callers may further filter if needed.
    """
    prefix: Optional[str] = None
    nums: List[int] = []
    for v in unique_sorted_strs:
        m = _PREFIX_HASH_INT_RE.fullmatch(v)
        if not m:
            return None
        p, num_str = m.group(1), m.group(2)
        # Reject zero-padded suffixes (e.g. Customer#000000001) – not a compact enum column
        if len(num_str) > 1 and num_str[0] == '0':
            return None
        n = int(num_str)
        if prefix is None:
            prefix = p
        elif prefix != p:
            return None  # mixed prefixes – fall back to full dict
        nums.append(n)
    if prefix is None:
        return None
    # Sort by str repr to preserve same order as alphabetical sort of full strings
    sorted_nums = sorted(set(nums), key=str)
    return prefix, sorted_nums


def _detect_prefix_hash_padded_int(
    unique_sorted_strs: List[str],
) -> Optional[Dict[str, Any]]:
    """Return compact meta if all values match PREFIX#ZERO_PADDED_N pattern.

    Zero-padded numbers have a consistent suffix width with a leading '0'.
    Returns: for contiguous ranges  {"prefix": ..., "pad": N, "min": M, "max": X}
             for sparse ranges      {"prefix": ..., "pad": N, "nums": [int ...]}
    """
    prefix: Optional[str] = None
    pad_width: Optional[int] = None
    nums: List[int] = []
    for v in unique_sorted_strs:
        hash_pos = v.rfind('#')
        if hash_pos < 0:
            return None
        p = v[:hash_pos + 1]          # includes the '#'
        num_str = v[hash_pos + 1:]
        if not num_str or not num_str.isdigit():
            return None
        # Must have an actual leading zero (zero-padding)
        if num_str[0] != '0':
            return None
        if prefix is None:
            prefix = p
            pad_width = len(num_str)
        elif prefix != p or len(num_str) != pad_width:
            return None  # inconsistent prefix or padding width
        nums.append(int(num_str))
    if prefix is None or pad_width is None or not nums:
        return None
    sorted_nums = sorted(set(nums))
    min_n, max_n = sorted_nums[0], sorted_nums[-1]
    # Contiguous range → store only bounds (most compact)
    if max_n - min_n == len(sorted_nums) - 1:
        return {"prefix": prefix, "pad": pad_width, "min": min_n, "max": max_n}
    # Sparse range → store bare int list (still far smaller than full strings)
    return {"prefix": prefix, "pad": pad_width, "nums": sorted_nums}


# ---------------------------------------------------------------------------
# Auto-encoding for a column's raw tokens
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _ColumnEncodingPlan:
    filter_by_token: Dict[str, int]
    enc_by_token: Dict[str, int]
    forward_map: Optional[Dict[str, int]]
    meta: Dict[str, Any]


def _plan_column_encoding(
    raw_tokens: Iterable[str],
    col_name: str,
) -> _ColumnEncodingPlan:
    """
    Auto-encode a list of raw string tokens from a .tbl column.

    Returns:
        filter_values  – values to write to {col}.bin
        enc_values     – values to write to {col}_enc.bin (0-based rank)
        forward_map    – {string → int} for string columns, None for int columns
        meta           – decode_meta entry dict
    """
    normalized_unique = sorted({_norm(t) for t in raw_tokens})

    # Try yearmonth format first (d_yearmonth column)
    if "yearmonth" in col_name and not "yearmonthnum" in col_name:
        ym_vals = [_try_yearmonth(t) for t in normalized_unique]
        if all(v is not None for v in ym_vals):
            values: List[int] = [v for v in ym_vals]  # type: ignore
            unique_sorted = sorted(set(values))
            rank = {v: i for i, v in enumerate(unique_sorted)}
            rev = {str(i): str(v) for i, v in enumerate(unique_sorted)}
            return _ColumnEncodingPlan(
                filter_by_token={token: value for token, value in zip(normalized_unique, values)},
                enc_by_token={token: rank[value] for token, value in zip(normalized_unique, values)},
                forward_map={token: rank[value] for token, value in zip(normalized_unique, values)},
                meta={"kind": "yearmonth", "reverse_map": rev},
            )

    # Try pure integer
    int_vals = [_try_int(t) for t in normalized_unique]
    if all(v is not None for v in int_vals):
        values = [v for v in int_vals]  # type: ignore
        unique_sorted = sorted(set(values))
        rank = {v: i for i, v in enumerate(unique_sorted)}
        rev = {str(i): str(v) for i, v in enumerate(unique_sorted)}
        return _ColumnEncodingPlan(
            filter_by_token={token: value for token, value in zip(normalized_unique, values)},
            enc_by_token={token: rank[value] for token, value in zip(normalized_unique, values)},
            forward_map={token: rank[value] for token, value in zip(normalized_unique, values)},
            meta={"kind": "int", "reverse_map": rev},
        )

    # String: dict encode (alphabetical sort → 0-based rank)
    unique_sorted_str = normalized_unique

    # Compact encoding for PREFIX#NUMBER columns (e.g. MFGR#1, MFGR#1101)
    phd = _detect_prefix_hash_int(unique_sorted_str)
    if phd is not None:
        prefix, sorted_nums = phd
        rank_map: Dict[str, int] = {prefix + str(n): i for i, n in enumerate(sorted_nums)}
        fwd_phi: Dict[str, int] = dict(rank_map)
        meta_phi: Dict[str, Any] = {"kind": "prefix_hash_int", "prefix": prefix, "nums": sorted_nums}
        return _ColumnEncodingPlan(
            filter_by_token=rank_map,
            enc_by_token=rank_map,
            forward_map=fwd_phi,
            meta=meta_phi,
        )

    # Compact encoding for PREFIX#ZERO_PADDED_N columns (e.g. Customer#000000001)
    phpd = _detect_prefix_hash_padded_int(unique_sorted_str)
    if phpd is not None:
        pfx = phpd["prefix"]
        pad = phpd["pad"]
        if "nums" in phpd:
            rank_map_p: Dict[str, int] = {pfx + str(n).zfill(pad): i for i, n in enumerate(phpd["nums"])}
        else:
            rank_map_p = {pfx + str(n).zfill(pad): i for i, n in enumerate(range(phpd["min"], phpd["max"] + 1))}
        meta_padded: Dict[str, Any] = {"kind": "prefix_hash_padded_int", **phpd}
        return _ColumnEncodingPlan(
            filter_by_token=rank_map_p,
            enc_by_token=rank_map_p,
            forward_map=dict(rank_map_p),
            meta=meta_padded,
        )

    # Fallback: full dict encoding (store complete reverse map)
    str_rank = {v: i for i, v in enumerate(unique_sorted_str)}
    fwd: Dict[str, int] = {v: i for i, v in enumerate(unique_sorted_str)}
    rev_str = {str(i): v for i, v in enumerate(unique_sorted_str)}
    return _ColumnEncodingPlan(
        filter_by_token=str_rank,
        enc_by_token=str_rank,
        forward_map=fwd,
        meta={"kind": "dict", "reverse_map": rev_str},
    )


def _encode_column(
    raw_tokens: List[str],
    col_name: str,
) -> Tuple[List[int], List[int], Optional[Dict[str, int]], Dict[str, Any]]:
    plan = _plan_column_encoding(raw_tokens, col_name)
    normalized = [_norm(t) for t in raw_tokens]
    return (
        [plan.filter_by_token[token] for token in normalized],
        [plan.enc_by_token[token] for token in normalized],
        plan.forward_map,
        plan.meta,
    )


# ---------------------------------------------------------------------------
# Lineorder conversion (fact table – all INTEGER columns)
# ---------------------------------------------------------------------------

def _convert_lineorder(
    input_dir: Path,
    output_dir: Path,
    ddl_col_order: List[str],
    extract_cols: List[str],
) -> int:
    src = input_dir / "lineorder.tbl"
    if not src.exists():
        raise FileNotFoundError(f"Missing: {src}")

    col_to_idx: Dict[str, int] = {c: i for i, c in enumerate(ddl_col_order) if c in extract_cols}
    files = {c: (output_dir / f"{c}.bin").open("wb") for c in extract_cols}
    count = 0
    try:
        for parts in iter_tbl_rows(src):
            for col in extract_cols:
                idx = col_to_idx.get(col)
                raw = parts[idx] if (idx is not None and idx < len(parts)) else ""
                v = _try_int(raw)
                files[col].write(struct.pack("<i", v if v is not None else 0))
            count += 1
            if count % 1_000_000 == 0:
                print(f"  lineorder_rows={count}")
    finally:
        for fp in files.values():
            fp.close()
    return count


# ---------------------------------------------------------------------------
# Dimension table conversion
# ---------------------------------------------------------------------------

def _convert_dimension(
    input_dir: Path,
    output_dir: Path,
    table: str,
    ddl_col_order: List[str],
    key_col: str,
) -> Tuple[int, Dict[str, Any]]:
    """
    Convert one SSB dimension table.

    key_col: the integer primary-key column (written as-is, no _enc).
    All other columns are auto-encoded (int → raw+rank, string → dict).

    Returns (row_count, table_meta).
    """
    src = input_dir / f"{table}.tbl"
    if not src.exists():
        print(f"  skip_missing={src}")
        return 0, {}

    col_to_idx: Dict[str, int] = {c: i for i, c in enumerate(ddl_col_order)}

    non_key_cols = [col for col in ddl_col_order if col != key_col and col in col_to_idx]
    unique_tokens_by_col: Dict[str, Set[str]] = {col: set() for col in non_key_cols}

    count = 0
    for parts in iter_tbl_rows(src):
        count += 1
        for col in non_key_cols:
            idx = col_to_idx[col]
            token = parts[idx] if idx < len(parts) else ""
            unique_tokens_by_col[col].add(_norm(token))

    if count == 0:
        return 0, {}

    table_meta: Dict[str, Any] = {}
    plans = {col: _plan_column_encoding(unique_tokens_by_col[col], col) for col in non_key_cols}

    key_file = None
    if key_col and key_col in col_to_idx:
        key_file = (output_dir / f"{key_col}.bin").open("wb")
        table_meta[key_col] = {"kind": "int"}

    # Decide _enc.bin emission using a generic cardinality threshold.
    # No query-specific analysis involved — purely data-driven.
    emit_groupby_enc_by_col: Dict[str, bool] = {}
    for col in non_key_cols:
        plan = plans[col]
        emit_groupby_enc = True
        if plan.meta.get("kind") in _STRING_ENCODING_KINDS:
            distinct_count = len(unique_tokens_by_col[col])
            emit_groupby_enc = (distinct_count <= _MAX_ENC_DISTINCT)
        emit_groupby_enc_by_col[col] = emit_groupby_enc

    value_files = {}
    for col in non_key_cols:
        enc_file = None
        if emit_groupby_enc_by_col[col]:
            enc_file = (output_dir / f"{col}_enc.bin").open("wb")
        value_files[col] = (
            (output_dir / f"{col}.bin").open("wb"),
            enc_file,
        )

    try:
        for parts in iter_tbl_rows(src):
            if key_file is not None:
                ki = col_to_idx[key_col]
                v = _try_int(parts[ki] if ki < len(parts) else "")
                key_file.write(struct.pack("<i", v if v is not None else 0))

            for col in non_key_cols:
                idx = col_to_idx[col]
                token = _norm(parts[idx] if idx < len(parts) else "")
                plan = plans[col]
                filter_file, enc_file = value_files[col]
                filter_file.write(struct.pack("<i", plan.filter_by_token[token]))
                if enc_file is not None:
                    enc_file.write(struct.pack("<i", plan.enc_by_token[token]))
    finally:
        if key_file is not None:
            key_file.close()
        for filter_file, enc_file in value_files.values():
            filter_file.close()
            if enc_file is not None:
                enc_file.close()

    for col in non_key_cols:
        plan = plans[col]
        if plan.forward_map is not None:
            (output_dir / f"{col}_dict.json").write_text(
                json.dumps(plan.forward_map, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        table_meta[col] = plan.meta
        if emit_groupby_enc_by_col[col]:
            table_meta[f"{col}_enc"] = {**plan.meta, "source_column": col}

    return count, table_meta


# ---------------------------------------------------------------------------
# Key column detection per table
# ---------------------------------------------------------------------------

_SSB_KEY_COLS: Dict[str, str] = {
    "date":     "d_datekey",
    "customer": "c_custkey",
    "supplier": "s_suppkey",
    "part":     "p_partkey",
}

# Lineorder integer columns to extract (all INTEGER columns in DDL)
_LINEORDER_EXTRACT_COLS = [
    "lo_orderkey", "lo_linenumber", "lo_custkey", "lo_partkey", "lo_suppkey",
    "lo_orderdate", "lo_shippriority", "lo_quantity", "lo_extendedprice",
    "lo_ordtotalprice", "lo_discount", "lo_revenue", "lo_supplycost",
    "lo_tax", "lo_commitdate",
]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare FaScalSQL SSB .bin columns from SSB .tbl data (DDL-driven, no policy file)"
    )
    parser.add_argument("--input-dir",   required=True, help="Directory with SSB .tbl files")
    parser.add_argument("--output-dir",  required=True, help="Output directory for .bin files")
    parser.add_argument("--meta-output", default=None,
                        help="Path for decode metadata JSON (default: <output-dir>/ssb_decode_meta.json)")
    args = parser.parse_args()

    input_dir  = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    meta_out = Path(args.meta_output) if args.meta_output else (output_dir / "ssb_decode_meta.json")

    repo_root = Path(__file__).resolve().parents[3]  # DMO-DB-FASCAL/
    ddl_path  = repo_root / "docs" / "schemas" / "ssb.ddl"
    if not ddl_path.exists():
        raise FileNotFoundError(f"SSB DDL not found: {ddl_path}")

    ddl_tables = _parse_ddl(ddl_path.read_text(encoding="utf-8"))

    meta_tables: Dict[str, Any] = {}

    # ---- Lineorder (fact table) ----
    lo_ddl_order = ddl_tables.get("lineorder", [])
    # Extract only the integer columns we know about (skip varchar lo_orderpriority etc.)
    extract = [c for c in _LINEORDER_EXTRACT_COLS if c in lo_ddl_order]
    lineorder_rows = _convert_lineorder(input_dir, output_dir, lo_ddl_order, extract)
    (output_dir / "lineorder_count.txt").write_text(str(lineorder_rows) + "\n")
    print(f"lineorder_rows={lineorder_rows}")
    meta_tables["lineorder"] = {c: {"kind": "int"} for c in extract}

    # ---- Dimension tables ----
    for table, key_col in _SSB_KEY_COLS.items():
        ddl_col_order = ddl_tables.get(table, [])
        if not ddl_col_order:
            print(f"  WARNING: table '{table}' not found in DDL, skipping")
            continue
        n, table_meta = _convert_dimension(
            input_dir, output_dir, table, ddl_col_order, key_col
        )
        if n > 0:
            print(f"{table}_rows={n}")
            (output_dir / f"{table}_count.txt").write_text(str(n) + "\n")
            meta_tables[table] = table_meta

    meta_payload: Dict[str, Any] = {
        "version": 2,
        "tables": meta_tables,
    }
    json_str = json.dumps(meta_payload, indent=2, ensure_ascii=False)

    # Primary output (explicit --meta-output or default <output-dir>/ssb_decode_meta.json)
    meta_out.parent.mkdir(parents=True, exist_ok=True)
    meta_out.write_text(json_str, encoding="utf-8")
    print(f"meta_output={meta_out}")

    # Always keep output_dir copy in sync (so runtime lookup at /tmp/.../ssb_decode_meta.json works)
    output_dir_meta = output_dir / "ssb_decode_meta.json"
    if meta_out.resolve() != output_dir_meta.resolve():
        output_dir_meta.write_text(json_str, encoding="utf-8")

    print(f"output_dir={output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
