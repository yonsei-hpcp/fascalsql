#!/usr/bin/env python3
"""
Prepare FaScalSQL TPC-H .bin column files from .tbl source data.

Fully DDL-driven: reads docs/schemas/tpch.ddl for column names, types, and order.
No external policy file required.  No query-specific computation.

Encoding is derived from DDL type:
  INTEGER              -> raw int32
  DECIMAL(p, s)        -> int32  (value x 10^s)
  DATE                 -> int32  YYYYMMDD
  CHAR(1)              -> int32  ord(character)
  CHAR(n), VARCHAR(n)  -> int32  dict-encoded (insertion-order; _enc.bin uses alphabetical rank)

Outputs per column:
  {col}.bin       - encoded int32 values (one per row)
  {col}_enc.bin   - 0-based dense rank (for GROUP BY hashtable sizing)
                    Emitted for all string columns with distinct_count <= 1000.
  {col}_dict.json - forward map {string -> int} for dict columns

Also writes:
  tpch_decode_meta.json  - column metadata and reverse maps

Note: derived columns (l_year, o_order_year, c_cntrycode) and attr-vs-attr
comparison flags (l_commit_before_receipt, l_ship_before_commit) are no longer
precomputed during data preparation.  They are computed at query time by the
lowering / codegen layer.
"""

import argparse
import json
import re
import struct
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# Generic cardinality threshold: dict columns with more distinct values
# than this do NOT get _enc.bin (too large for hash-table GROUP BY).
# This is a data-only heuristic, not driven by any specific query.
_MAX_ENC_DISTINCT = 1000

# Dict JSON generation threshold: dict columns with more distinct values
# than this do NOT get a *_dict.json file.  The Python codegen layer uses
# dict JSON to resolve LIKE/NOT_LIKE predicates at codegen time (by iterating
# all entries to find matches).  For columns exceeding this threshold the
# resolution is deferred to the C++ runtime which reads the compact binary
# dict files instead.  The binary files (_dict_strings.bin / _dict_offsets.bin)
# are always written regardless of cardinality.
_MAX_DICT_JSON_DISTINCT = 100_000

# No query-specific extra columns any more.
_TABLE_EXTRA_INT_COLS: Dict[str, Tuple[str, ...]] = {}

_TABLE_EXTRA_FLOAT_COLS: Dict[str, Tuple[str, ...]] = {}


# ---------------------------------------------------------------------------
# DDL parsing
# ---------------------------------------------------------------------------

_CREATE_TABLE_RE = re.compile(
    r"CREATE\s+TABLE\s+([A-Za-z_]\w*)\s*\((.*?)\)\s*;",
    re.IGNORECASE | re.DOTALL,
)
_COLUMN_RE = re.compile(
    r"^\s*([A-Za-z_]\w*)\s+([A-Za-z]+(?:\([^)]*\))?)",
    re.IGNORECASE,
)
_CONSTRAINT_KEYWORDS = {"primary", "foreign", "constraint", "unique", "check"}


def _strip_comments(text: str) -> str:
    return "\n".join(line.split("--", 1)[0] for line in text.splitlines())


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


def _parse_ddl(ddl_text: str) -> Dict[str, Dict[str, str]]:
    """Return {table -> {col -> sql_type}} preserving DDL column order."""
    tables: Dict[str, Dict[str, str]] = {}
    for m in _CREATE_TABLE_RE.finditer(_strip_comments(ddl_text)):
        tname = m.group(1).strip().lower()
        cols: Dict[str, str] = {}
        for item in _split_top_level(m.group(2)):
            item = item.strip()
            if not item:
                continue
            first = item.split()[0].lower() if item.split() else ""
            if first in _CONSTRAINT_KEYWORDS:
                continue
            cm = _COLUMN_RE.match(item)
            if cm:
                cols[cm.group(1).strip().lower()] = cm.group(2).strip().upper()
        tables[tname] = cols
    return tables


def _decimal_scale(sql_type: str) -> Optional[int]:
    """Return the decimal scale s for DECIMAL(p,s), or None."""
    m = re.match(r"(?:DECIMAL|NUMERIC)\s*\(\s*\d+\s*,\s*(\d+)\s*\)", sql_type, re.IGNORECASE)
    if m:
        return int(m.group(1))
    if re.match(r"(?:DECIMAL|NUMERIC)$", sql_type, re.IGNORECASE):
        return 2
    return None


# Only the lineitem fact-table decimal columns are stored as IEEE-754 float32.
# Dimension-table decimal columns (c_acctbal, o_totalprice, ps_supplycost, etc.)
# remain as scaled_decimal (int32 * 10^scale) to avoid HT payload reinterpretation.
_FLOAT_DECIMAL_TABLES: Set[str] = {"lineitem"}


def _ddl_kind(sql_type: str, table: str = "") -> str:
    """Encoding kind from DDL SQL type."""
    t = sql_type.strip().upper()
    if re.match(r"(?:DECIMAL|NUMERIC)", t):
        if table in _FLOAT_DECIMAL_TABLES:
            return "float_decimal"  # lineitem only: store as IEEE-754 float32
        return "scaled_decimal"  # dim tables: store as scaled int32
    if t.startswith("DATE"):
        return "date"
    if re.fullmatch(r"CHAR\s*\(\s*1\s*\)", t):
        return "char_code"
    if re.match(r"(?:CHAR|VARCHAR)", t):
        return "dict"
    return "int"


# ---------------------------------------------------------------------------
# .tbl parsing helpers
# ---------------------------------------------------------------------------

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _parse_tbl_line(line: str) -> List[str]:
    parts = line.rstrip("\n").split("|")
    if parts and parts[-1] == "":
        parts = parts[:-1]
    return parts


def _encode_value(raw: str, kind: str, scale: int, fwd_map: Dict[str, int]) -> Any:
    """Convert raw .tbl token to encoded value (int32 for most kinds, float32 for float_decimal)."""
    if kind == "dict":
        # Preserve leading/trailing spaces — they are part of the string value
        t = raw
    else:
        t = raw.strip().strip('"')
    if not t:
        return 0.0 if kind == "float_decimal" else 0
    if kind == "int":
        try:
            return int(t)
        except ValueError:
            return 0
    if kind == "float_decimal":
        try:
            return float(t)
        except ValueError:
            return 0.0
    if kind == "scaled_decimal":  # backward compat only
        try:
            if "." in t:
                return int(round(float(t) * scale))
            return int(t) * scale
        except ValueError:
            return 0
    if kind == "date":
        return int(t.replace("-", "")) if _DATE_RE.match(t) else 0
    if kind == "char_code":
        return ord(t[0]) if t else 0
    if kind == "dict":
        if t not in fwd_map:
            fwd_map[t] = len(fwd_map)
        return fwd_map[t]
    return 0


# Stub evaluators for table-extra columns (no-op since dicts are empty).
def _eval_table_extra_int(
    table: str,
    col: str,
    row_vals: Dict[str, Any],
    ctx: Dict[str, Any],
) -> int:
    return 0


def _eval_table_extra_float(
    table: str,
    col: str,
    row_vals: Dict[str, Any],
    ctx: Dict[str, Any],
) -> float:
    return 0.0


# ---------------------------------------------------------------------------
# _enc.bin re-ranking
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# PREFIX#NUMBER compact encoding detection (e.g. Manufacturer#1, Brand#11)
# ---------------------------------------------------------------------------

_PREFIX_HASH_INT_RE = re.compile(r'^(.+#)(\d+)$')


def _detect_prefix_hash_int(
    unique_sorted_strs: List[str],
) -> Optional[Tuple[str, List[int]]]:
    """Return (prefix, nums) if all values match PREFIX#INTEGER pattern.

    nums is sorted by str() representation of the integer suffix, which
    matches the alphabetical sort order of the full strings (same prefix).
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
            return None
        nums.append(n)
    if prefix is None:
        return None
    sorted_nums = sorted(set(nums), key=str)
    return prefix, sorted_nums


def _detect_prefix_hash_padded_int(
    unique_sorted_strs: List[str],
) -> Optional[Dict[str, Any]]:
    """Return compact meta if all values match PREFIX#ZERO_PADDED_N pattern.

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
        p = v[:hash_pos + 1]
        num_str = v[hash_pos + 1:]
        if not num_str or not num_str.isdigit():
            return None
        if num_str[0] != '0':
            return None
        if prefix is None:
            prefix = p
            pad_width = len(num_str)
        elif prefix != p or len(num_str) != pad_width:
            return None
        nums.append(int(num_str))
    if prefix is None or pad_width is None or not nums:
        return None
    sorted_nums = sorted(set(nums))
    min_n, max_n = sorted_nums[0], sorted_nums[-1]
    if max_n - min_n == len(sorted_nums) - 1:
        return {"prefix": prefix, "pad": pad_width, "min": min_n, "max": max_n}
    return {"prefix": prefix, "pad": pad_width, "nums": sorted_nums}


def _rerank_enc(
    enc_path: Path,
    kind: str,
    fwd_map: Dict[str, int],
) -> Dict[str, str]:
    """
    Re-rank the raw values in enc_path to 0-based dense integers.
    For dict kind: sort alphabetically.
    For other kinds: sort numerically.
    Returns reverse_map {encoded_str -> original_str} for decode_meta.
    """
    raw_vals: List[int] = []
    with enc_path.open("rb") as f:
        while True:
            chunk = f.read(4)
            if not chunk:
                break
            raw_vals.append(struct.unpack("<i", chunk)[0])

    unique_sorted = sorted(set(raw_vals))
    rank_map = {v: i for i, v in enumerate(unique_sorted)}

    if kind == "dict":
        # Rebuild alphabetical rank from forward map
        inv = {v: k for k, v in fwd_map.items()}
        alpha_keys = sorted(inv.values())
        alpha_rank = {k: i for i, k in enumerate(alpha_keys)}
        alpha_fwd  = {k: i for i, k in enumerate(alpha_keys)}
        # Remap: raw insertion-order id -> alphabetical rank
        insertion_to_alpha = {fwd_map[k]: alpha_fwd[k] for k in fwd_map}
        with enc_path.open("wb") as f:
            for v in raw_vals:
                f.write(struct.pack("<i", insertion_to_alpha.get(v, v)))
        rev = {str(i): k for i, k in enumerate(alpha_keys)}
    elif kind == "char_code":
        with enc_path.open("wb") as f:
            for v in raw_vals:
                f.write(struct.pack("<i", rank_map[v]))
        rev = {str(rank): chr(v) for v, rank in rank_map.items()}
    else:
        with enc_path.open("wb") as f:
            for v in raw_vals:
                f.write(struct.pack("<i", rank_map[v]))
        rev = {str(rank): str(v) for v, rank in rank_map.items()}

    return rev


# ---------------------------------------------------------------------------
# Table conversion
# ---------------------------------------------------------------------------

def _convert_table(
    input_dir: Path,
    output_dir: Path,
    table: str,
    ddl_cols: Dict[str, str],   # {col -> sql_type}, preserves DDL order
    ctx: Dict[str, Any],
) -> Tuple[int, Dict[str, Any]]:
    """
    Convert one TPC-H table to .bin files.
    Returns (row_count, table_meta) for decode_meta.json.
    """
    src = input_dir / f"{table}.tbl"
    if not src.exists():
        print(f"  skip_missing={src}")
        return 0, {}
    output_dir.mkdir(parents=True, exist_ok=True)

    # All DDL columns are physical (no precomputed columns any more).
    real_cols = list(ddl_cols.keys())
    tbl_col_order = list(ddl_cols.keys())

    # Encoding metadata per column
    col_kind:  Dict[str, str] = {}
    col_scale: Dict[str, int] = {}
    for col in real_cols:
        kind = _ddl_kind(ddl_cols[col], table)
        col_kind[col] = kind
        if kind == "scaled_decimal":
            s = _decimal_scale(ddl_cols[col])
            col_scale[col] = 10 ** (s if s is not None else 2)
        else:
            col_scale[col] = 1

    # Determine which cols get _enc.bin.
    # Rule: dict/char_code/date columns always; INTEGER columns in non-lineitem tables.
    # float_decimal and scaled_decimal columns: no rank-encoding needed (continuous values).
    # Dict columns: _enc.bin is emitted unconditionally here and pruned
    # post-hoc based on the generic cardinality threshold _MAX_ENC_DISTINCT.
    enc_cols: Set[str] = set()
    dict_cols_to_prune: Set[str] = set()
    for col in real_cols:
        if col_kind[col] in ("float_decimal", "scaled_decimal"):
            pass  # no _enc.bin for decimal columns
        elif col_kind[col] == "dict":
            enc_cols.add(col)
            dict_cols_to_prune.add(col)
        elif col_kind[col] != "int":
            enc_cols.add(col)
        elif table != "lineitem":
            enc_cols.add(col)

    table_extra_int_cols = list(_TABLE_EXTRA_INT_COLS.get(table, ()))
    table_extra_float_cols = list(_TABLE_EXTRA_FLOAT_COLS.get(table, ()))

    # Open output files
    fps: Dict[str, Any] = {}
    for col in real_cols:
        fps[col] = (output_dir / f"{col}.bin").open("wb")
    for extra_col in table_extra_int_cols:
        fps[extra_col] = (output_dir / f"{extra_col}.bin").open("wb")
    for extra_col in table_extra_float_cols:
        fps[extra_col] = (output_dir / f"{extra_col}.bin").open("wb")
    for col in enc_cols:
        fps[f"{col}_enc"] = (output_dir / f"{col}_enc.bin").open("wb")

    # Per-column state
    fwd_maps: Dict[str, Dict[str, int]] = {c: {} for c in real_cols if col_kind[c] == "dict"}
    rows = 0

    try:
        with src.open(encoding="utf-8") as fin:
            for line in fin:
                parts = _parse_tbl_line(line)
                row_vals: Dict[str, Any] = {}  # may hold int or float
                raw_vals: Dict[str, str] = {}

                for col in real_cols:
                    tbl_idx = tbl_col_order.index(col)
                    raw = parts[tbl_idx] if tbl_idx < len(parts) else ""
                    raw_vals[col] = raw
                    fwd = fwd_maps.get(col, {})
                    v = _encode_value(raw, col_kind[col], col_scale[col], fwd)
                    row_vals[col] = v
                    if col_kind[col] == "float_decimal":
                        fps[col].write(struct.pack("<f", float(v)))  # IEEE-754 float32
                    else:
                        fps[col].write(struct.pack("<i", int(v)))
                    if col in enc_cols:
                        fps[f"{col}_enc"].write(struct.pack("<i", int(v)))

                for extra_col in table_extra_int_cols:
                    extra_val = _eval_table_extra_int(table, extra_col, row_vals, ctx)
                    row_vals[extra_col] = extra_val
                    fps[extra_col].write(struct.pack("<i", extra_val))
                for extra_col in table_extra_float_cols:
                    extra_val = _eval_table_extra_float(table, extra_col, row_vals, ctx)
                    row_vals[extra_col] = extra_val
                    fps[extra_col].write(struct.pack("<f", float(extra_val)))

                rows += 1
                if rows % 2_000_000 == 0:
                    print(f"  {table}_rows={rows}")
    finally:
        for fp in fps.values():
            fp.close()

    # Prune _enc.bin for dict columns exceeding the generic cardinality threshold.
    # No query-specific analysis involved — purely data-driven.
    final_enc_cols = set(enc_cols)
    for col in dict_cols_to_prune:
        distinct_count = len(fwd_maps.get(col, {}))
        if distinct_count > _MAX_ENC_DISTINCT:
            final_enc_cols.discard(col)
            enc_path = output_dir / f"{col}_enc.bin"
            if enc_path.exists():
                enc_path.unlink()

    # Re-rank _enc.bin files and build table_meta
    table_meta: Dict[str, Any] = {}

    for col in real_cols:
        kind = col_kind[col]
        scale_val = _decimal_scale(ddl_cols[col])
        fwd = fwd_maps.get(col, {})

        if kind == "float_decimal":
            table_meta[col] = {"kind": "float_decimal"}
        elif kind == "scaled_decimal":  # backward compat
            s = scale_val if scale_val is not None else 2
            table_meta[col] = {"kind": f"scaled_decimal_{s}", "scale": s}
        elif kind == "dict":
            alpha_keys = sorted(fwd.keys())
            alpha_fwd = {k: i for i, k in enumerate(alpha_keys)}
            distinct_count = len(alpha_keys)
            is_high_cardinality = distinct_count > _MAX_DICT_JSON_DISTINCT

            if not is_high_cardinality:
                # Low-cardinality: write dict JSON (small file, safe to load at codegen time)
                (output_dir / f"{col}_dict.json").write_text(
                    json.dumps(alpha_fwd, ensure_ascii=False, indent=2), encoding="utf-8"
                )
            else:
                # High-cardinality: do NOT write dict JSON (would be huge at SF>=10).
                # Remove any stale dict JSON from a previous run.
                stale_json = output_dir / f"{col}_dict.json"
                if stale_json.exists():
                    stale_json.unlink()
                    print(f"  removed stale high-cardinality dict JSON: {stale_json.name}")

            # Always write _dict_strings.bin + _dict_offsets.bin for runtime LIKE support.
            # These are binary packed and loaded by the C++ runtime (not by Python codegen).
            _write_dict_binary_files(output_dir, table, col, alpha_fwd)

            _rewrite_bin_to_alpha(output_dir / f"{col}.bin", fwd)
            if col in final_enc_cols:
                _rerank_enc(output_dir / f"{col}_enc.bin", "dict", fwd)
            rev_alpha = {str(i): raw for i, raw in enumerate(alpha_keys)}
            # Use compact PREFIX#NUMBER metadata when all values match the pattern
            phd = _detect_prefix_hash_int(alpha_keys)
            if phd is not None:
                prefix, sorted_nums = phd
                table_meta[col] = {"kind": "prefix_hash_int", "prefix": prefix, "nums": sorted_nums}
            else:
                # Try zero-padded variant: e.g. Customer#000000001
                phpd = _detect_prefix_hash_padded_int(alpha_keys)
                if phpd is not None:
                    table_meta[col] = {"kind": "prefix_hash_padded_int", **phpd}
                else:
                    meta_entry: Dict[str, Any] = {"kind": "dict", "distinct_count": distinct_count}
                    if not is_high_cardinality and len(rev_alpha) <= 100:
                        meta_entry["reverse_map"] = rev_alpha
                    table_meta[col] = meta_entry
        elif kind == "char_code":
            if col in final_enc_cols:
                rev = _rerank_enc(output_dir / f"{col}_enc.bin", "char_code", {})
                table_meta[col] = {"kind": "char_code", "reverse_map": rev}
            else:
                table_meta[col] = {"kind": "char_code"}
        elif kind == "date":
            if col in final_enc_cols:
                rev = _rerank_enc(output_dir / f"{col}_enc.bin", "date", {})
                if len(rev) <= 200:
                    table_meta[col] = {"kind": "date", "reverse_map": rev}
                else:
                    table_meta[col] = {"kind": "date"}
            else:
                table_meta[col] = {"kind": "date"}
        else:  # int
            if col in final_enc_cols:
                rev = _rerank_enc(output_dir / f"{col}_enc.bin", "int", {})
                if len(rev) <= 200:
                    table_meta[col] = {"kind": "int", "reverse_map": rev}
                else:
                    table_meta[col] = {"kind": "int"}
            else:
                table_meta[col] = {"kind": "int"}

    for extra_col in table_extra_int_cols:
        table_meta[extra_col] = {"kind": "int"}
    for extra_col in table_extra_float_cols:
        table_meta[extra_col] = {"kind": "float_decimal"}

    return rows, table_meta


def _write_dict_binary_files(
    output_dir: Path,
    table: str,
    column: str,
    alpha_fwd: Dict[str, int],
) -> None:
    """Write *_dict_strings.bin* and *_dict_offsets.bin* for a dict-encoded column.

    These compact binary files are loaded by the C++ runtime to support GPU LIKE
    predicate evaluation via ``regex_bitset_kernel``.  Writing them at data-
    preparation time (instead of at codegen time) removes the need for the Python
    codegen layer to ever load potentially huge dict JSON files.

    Layout:
        *_dict_strings.bin*  -- concatenated null-terminated UTF-8 strings sorted
                                by encoded ID.
        *_dict_offsets.bin*  -- int32 array of byte offsets into the strings blob,
                                one per encoded ID.
    """
    if not alpha_fwd:
        return
    max_id = max(alpha_fwd.values())
    dict_size = max_id + 1
    sorted_entries = sorted(alpha_fwd.items(), key=lambda x: x[1])

    offsets = [0] * dict_size
    strings_parts: List[bytes] = []
    current_offset = 0
    for raw_val, enc_id in sorted_entries:
        if enc_id < 0 or enc_id >= dict_size:
            continue
        offsets[enc_id] = current_offset
        s_bytes = str(raw_val).encode("utf-8") + b"\0"
        strings_parts.append(s_bytes)
        current_offset += len(s_bytes)

    strings_bin = b"".join(strings_parts)
    offsets_bin = struct.pack(f"<{len(offsets)}i", *offsets)

    (output_dir / f"{table}_{column}_dict_strings.bin").write_bytes(strings_bin)
    (output_dir / f"{table}_{column}_dict_offsets.bin").write_bytes(offsets_bin)


def _rewrite_bin_to_alpha(bin_path: Path, fwd_map: Dict[str, int]) -> None:
    """Rewrite a dict .bin file to use alphabetical (sorted) rank instead of insertion order."""
    alpha_keys = sorted(fwd_map.keys())
    alpha_rank = {k: i for i, k in enumerate(alpha_keys)}
    insertion_to_alpha = {fwd_map[k]: alpha_rank[k] for k in fwd_map}

    raw_vals: List[int] = []
    with bin_path.open("rb") as f:
        while True:
            chunk = f.read(4)
            if not chunk:
                break
            raw_vals.append(struct.unpack("<i", chunk)[0])

    with bin_path.open("wb") as f:
        for v in raw_vals:
            f.write(struct.pack("<i", insertion_to_alpha.get(v, v)))


# ---------------------------------------------------------------------------
# FK encoding: generate {fk_col}_enc.bin for direct-address HT joins
# ---------------------------------------------------------------------------

def _generate_fk_enc(output_dir: Path, fk_col: str, pk_col: str) -> None:
    """Generate {fk_col}_enc.bin: map each fact-table row to 0-based dim row index.

    Reads {pk_col}.bin (dimension PK column) to build a key->row_index lookup,
    then reads {fk_col}.bin (fact FK column) and writes the mapped row indices.
    """
    import numpy as np

    pk_path = output_dir / f"{pk_col}.bin"
    fk_path = output_dir / f"{fk_col}.bin"
    out_path = output_dir / f"{fk_col}_enc.bin"

    if not pk_path.exists() or not fk_path.exists():
        print(f"  skip FK enc {fk_col}: missing {pk_path} or {fk_path}")
        return

    pk_vals = np.fromfile(str(pk_path), dtype=np.int32)
    fk_vals = np.fromfile(str(fk_path), dtype=np.int32)

    # Build PK value -> 0-based row index lookup
    pk_lookup: Dict[int, int] = {}
    for idx, v in enumerate(pk_vals):
        pk_lookup[int(v)] = idx

    # Map FK values to dimension row indices
    enc_vals = np.empty(len(fk_vals), dtype=np.int32)
    unmapped = 0
    for i, fk in enumerate(fk_vals):
        row_idx = pk_lookup.get(int(fk), -1)
        enc_vals[i] = row_idx
        if row_idx < 0:
            unmapped += 1

    enc_vals.tofile(str(out_path))
    print(f"  FK enc: {fk_col}_enc.bin ({len(fk_vals)} rows, {unmapped} unmapped)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare FaScalSQL TPC-H .bin columns from TPC-H .tbl data (DDL-driven)"
    )
    parser.add_argument("--input-dir",   required=True, help="Directory with TPC-H .tbl files")
    parser.add_argument("--output-dir",  required=True, help="Output directory for .bin files")
    parser.add_argument("--meta-output", default=None,
                        help="Path for decode metadata JSON (default: <output-dir>/tpch_decode_meta.json)")
    args = parser.parse_args()

    input_dir  = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    meta_out = Path(args.meta_output) if args.meta_output else (output_dir / "tpch_decode_meta.json")

    repo_root = Path(__file__).resolve().parents[3]  # DMO-DB-FASCAL/
    ddl_path  = repo_root / "docs" / "schemas" / "tpch.ddl"
    if not ddl_path.exists():
        raise FileNotFoundError(f"TPC-H DDL not found: {ddl_path}")

    ddl_tables = _parse_ddl(ddl_path.read_text(encoding="utf-8"))

    # No cross-table context needed — all derived columns are computed at query time.
    ctx: Dict[str, Any] = {}

    meta_tables: Dict[str, Any] = {}
    lineitem_rows = 0

    # Convert all tables listed in the DDL
    for table, ddl_cols in ddl_tables.items():
        n, table_meta = _convert_table(input_dir, output_dir, table, ddl_cols, ctx)
        if n > 0:
            print(f"{table}_rows={n}")
            (output_dir / f"{table}_count.txt").write_text(str(n) + "\n")
            meta_tables[table] = table_meta
            if table == "lineitem":
                lineitem_rows = n

    if lineitem_rows == 0:
        # lineitem_count.txt is the canonical row-count file for the engine
        pass
    else:
        (output_dir / "lineitem_count.txt").write_text(str(lineitem_rows) + "\n")

    # -----------------------------------------------------------------------
    # FK encoding: l_orderkey_enc.bin
    # Maps each lineitem row -> 0-based orders row index for direct-address HT.
    # -----------------------------------------------------------------------
    _generate_fk_enc(output_dir, "l_orderkey", "o_orderkey")

    meta_payload: Dict[str, Any] = {
        "version": 2,
        "tables": meta_tables,
    }
    json_str = json.dumps(meta_payload, indent=2, ensure_ascii=False)

    # Primary output (explicit --meta-output or default <output-dir>/tpch_decode_meta.json)
    meta_out.parent.mkdir(parents=True, exist_ok=True)
    meta_out.write_text(json_str, encoding="utf-8")
    print(f"meta_output={meta_out}")

    # Always keep output_dir copy in sync (so runtime lookup at /tmp/.../tpch_decode_meta.json works)
    output_dir_meta = output_dir / "tpch_decode_meta.json"
    if meta_out.resolve() != output_dir_meta.resolve():
        output_dir_meta.write_text(json_str, encoding="utf-8")

    print(f"output_dir={output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
