#!/usr/bin/env python3
"""
String Encoding Catalog for FaScalSQL

Single source of truth for all column encoding metadata.  Populated at data-
preparation time (prepare_fascalsql_tpch_bin.py / prepare_fascalsql_ssb_bin.py)
via decode_meta.json files.

Exposed API
===========
    catalog = load_encoding_catalog()

    # string → encoded int (dict columns)
    code = catalog.encode("nation", "n_name", "FRANCE")   # → 6

    # scale factor for DECIMAL columns (stored as value × scale)
    scale = catalog.is_scaled("lineitem", "l_extendedprice")  # → 100 or None

    # Attr-vs-Attr predicate → precomputed column (deprecated: no longer used)
    res = catalog.get_attr_compare_precomputed("l_commitdate", "<", "l_receiptdate")
    # → None (attr-vs-attr comparisons are now handled at query time)
"""

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# IR expression-type → SQL operator mapping (for policy JSON ↔ IR lookups)
# ---------------------------------------------------------------------------
_IR_TYPE_TO_SQL_OP: Dict[str, str] = {
    "SmallerExpr":      "<",
    "LargerExpr":       ">",
    "SmallerEqualExpr": "<=",
    "LargerEqualExpr":  ">=",
    "EqualsExpr":       "==",
}


@dataclass
class EncodingCatalog:
    # (table, col) → {raw_string → encoded_int}  (dict-encoded columns)
    forward: Dict[Tuple[str, str], Dict[str, int]]

    # (table, col) → metadata for externally stored forward maps; loaded on demand
    deferred_forward_meta: Dict[Tuple[str, str], Dict[str, Any]] = field(default_factory=dict)

    # (table, col) → scale multiplier (e.g. 100 for DECIMAL(15,2))
    scale_factors: Dict[Tuple[str, str], int] = field(default_factory=dict)

    # (left_col, sql_op, right_col) → (table, precomputed_col, value)
    attr_compare_precomputed: Dict[Tuple[str, str, str], Tuple[str, str, int]] = field(
        default_factory=dict
    )

    # (table, col) pairs whose binary file stores ASCII ordinals (CHAR(1) type);
    # predicates must reference the "_enc" companion file (compact rank 0,1,2,...)
    char_code_columns: Set[Tuple[str, str]] = field(default_factory=set)

    # (table, col) pairs whose binary file stores raw IEEE-754 float32 (not scaled int32)
    float_decimal_columns: Set[Tuple[str, str]] = field(default_factory=set)

    # Raw decode metadata per column, normalized as (table, col) -> metadata dict
    column_meta: Dict[Tuple[str, str], Dict[str, Any]] = field(default_factory=dict)

    @dataclass(frozen=True)
    class LikePredicatePlan:
        mode: str
        target_ids: Tuple[int, ...]
        source_column: str
        distinct_count: int

    # ---------------------------------------------------------------------------
    # Core API
    # ---------------------------------------------------------------------------

    def _ensure_forward_map(self, table: str, column: str) -> Optional[Dict[str, int]]:
        tname = _norm(table)
        cname = _norm(column)
        resolved = self.resolve_source_column(tname, cname)
        keys_to_try = []
        for key in ((tname, resolved), (tname, cname)):
            if key not in keys_to_try:
                keys_to_try.append(key)

        for key in keys_to_try:
            mapping = self.forward.get(key)
            meta = self.deferred_forward_meta.get(key)
            # When both an inline forward map (from one decode_meta family)
            # and a deferred entry (from another family) exist for the same
            # (table, column) key, prefer the deferred lazy-load — it will
            # resolve from the correct family's data directory.
            if mapping is not None and meta is None:
                return mapping

            if meta is None:
                continue

            loaded = _load_external_forward_dict(key[0], key[1], meta)
            if not loaded:
                return None

            self.forward[key] = loaded
            if key != (tname, resolved):
                self.forward.setdefault((tname, resolved), loaded)
            if key != (tname, cname):
                self.forward.setdefault((tname, cname), loaded)
            src = meta.get("source_column")
            if isinstance(src, str) and src.strip():
                self.forward.setdefault((tname, _norm(src)), loaded)
            return loaded

        return None

    def encode(self, table: str, column: str, raw_value: str) -> Optional[int]:
        """Return the encoded integer for *raw_value* in *table.column*, or None."""
        mapping = self._ensure_forward_map(table, column)
        if mapping is None:
            return None
        return mapping.get(str(raw_value))

    def is_scaled(self, table: str, column: str) -> Optional[int]:
        """Return the storage scale (e.g. 100) if *column* is a scaled DECIMAL, else None."""
        return self.scale_factors.get((_norm(table), _norm(column)))

    def is_float_decimal(self, table: str, column: str) -> bool:
        """Return True if *column* in *table* is stored as IEEE-754 float32 (not scaled int32)."""
        return (_norm(table), _norm(column)) in self.float_decimal_columns

    def is_char_code(self, table: str, column: str) -> bool:
        """Return True if *column* in *table* is CHAR(1) / char_code kind.

        For such columns the plain *.bin* file stores raw ASCII ordinals while
        the *_enc.bin* companion stores compact rank values that match the
        integer predicates produced by the query planner.
        """
        return (_norm(table), _norm(column)) in self.char_code_columns

    def get_attr_compare_precomputed(
        self, left_col: str, op: str, right_col: str
    ) -> Optional[Tuple[str, str, int]]:
        """
        Look up a precomputed column for an Attr-vs-Attr comparison.

        *op* can be either a SQL operator ("<", ">", "<=", ">=", "==")
        or an IR expression type ("SmallerExpr", "LargerExpr", ...).
        Returns (table, precomputed_col, value) or None.
        """
        sql_op = _IR_TYPE_TO_SQL_OP.get(op, op)
        return self.attr_compare_precomputed.get(
            (_norm(left_col), sql_op, _norm(right_col))
        )

    def get_meta(self, table: str, column: str) -> Optional[Dict[str, Any]]:
        """Return raw decode metadata for *table.column* if available."""
        return self.column_meta.get((_norm(table), _norm(column)))

    def resolve_source_column(self, table: str, column: str) -> str:
        """Return the physical/source column that owns dictionary storage for *column*."""
        tname = _norm(table)
        cname = _norm(column)
        meta = self.column_meta.get((tname, cname), {})
        src = meta.get("source_column")
        if isinstance(src, str) and src.strip():
            return _norm(src)
        if cname.endswith("_enc"):
            base = cname[:-4]
            if (tname, base) in self.forward or (tname, base) in self.column_meta:
                return base
        return cname

    def get_forward_map(self, table: str, column: str) -> Optional[Dict[str, int]]:
        """Return the resolved forward map for a string-like encoded column if available."""
        return self._ensure_forward_map(table, column)

    def plan_like_predicate(
        self,
        table: str,
        column: str,
        pattern: str,
        *,
        is_not: bool = False,
        bitset_threshold: int = 100,
    ) -> Optional["EncodingCatalog.LikePredicatePlan"]:
        """Plan LIKE/NOT LIKE execution against the resolved string dictionary contract."""
        mapping = self.get_forward_map(table, column)
        if not mapping:
            return None

        source_column = self.resolve_source_column(table, column)
        matching_ids = sorted(
            {
                int(enc_id)
                for raw_value, enc_id in mapping.items()
                if _sql_like_match(pattern, raw_value)
            }
        )
        all_ids = sorted({int(enc_id) for enc_id in mapping.values()})
        distinct_count = len(all_ids)

        use_ne_and = False
        if is_not:
            if len(matching_ids) <= distinct_count / 2:
                use_ne_and = True
                target_ids = matching_ids
            else:
                target_ids = sorted(set(all_ids) - set(matching_ids))
        else:
            if len(matching_ids) <= distinct_count / 2:
                target_ids = matching_ids
            else:
                use_ne_and = True
                target_ids = sorted(set(all_ids) - set(matching_ids))

        if not target_ids:
            mode = "always_true" if use_ne_and else "always_false"
            return self.LikePredicatePlan(mode, tuple(), source_column, distinct_count)
        if len(target_ids) > bitset_threshold:
            return self.LikePredicatePlan("gpu_bitset", tuple(target_ids), source_column, distinct_count)
        if use_ne_and:
            return self.LikePredicatePlan("and_ne", tuple(target_ids), source_column, distinct_count)
        return self.LikePredicatePlan("or_eq", tuple(target_ids), source_column, distinct_count)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _norm(s: str) -> str:
    return str(s).strip().lower()


def _sql_like_match(pattern: str, value: str) -> bool:
    regex = ""
    for char in str(pattern):
        if char == "%":
            regex += ".*"
        elif char == "_":
            regex += "."
        else:
            regex += re.escape(char)
    return re.fullmatch(regex, str(value), re.DOTALL) is not None


def _candidate_runtime_data_dirs() -> list[Path]:
    repo_root = Path(__file__).resolve().parent.parent.parent.parent
    dirs: list[Path] = []
    env_dir = os.environ.get("FASCALSQL_DATA_DIR")
    if env_dir:
        dirs.append(Path(env_dir))
    dirs.extend(
        [
            repo_root / "tmp" / "fascalsql_tpch_data",
            repo_root / "tmp" / "fascalsql_ssb_data",
            repo_root / "tmp" / "fascalsql_data",
        ]
    )

    unique_dirs: list[Path] = []
    seen: set[str] = set()
    for d in dirs:
        key = str(d)
        if key in seen:
            continue
        seen.add(key)
        unique_dirs.append(d)
    return unique_dirs


def _load_external_forward_dict(table: str, column: str, meta: Dict[str, Any]) -> Optional[Dict[str, int]]:
    json_candidates = [f"{column}_dict.json"]
    src = meta.get("source_column")
    if isinstance(src, str) and src.strip():
        json_candidates.append(f"{_norm(src)}_dict.json")

    for data_dir in _candidate_runtime_data_dirs():
        for filename in json_candidates:
            path = data_dir / filename
            if not path.exists():
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if not isinstance(payload, dict):
                continue
            fwd: Dict[str, int] = {}
            for raw, enc in payload.items():
                try:
                    fwd[str(raw)] = int(enc)
                except Exception:
                    continue
            if fwd:
                return fwd

        dict_path = data_dir / f"{table}_{column}.dict"
        if not dict_path.exists():
            continue
        fwd = {}
        try:
            for line in dict_path.read_text(encoding="utf-8").splitlines():
                if "|" not in line:
                    continue
                enc_str, raw = line.split("|", 1)
                fwd[raw] = int(enc_str)
        except Exception:
            continue
        if fwd:
            return fwd

    return None


# ---------------------------------------------------------------------------
# decode_meta.json loader
# ---------------------------------------------------------------------------

def _load_one_meta(path: Path) -> Dict:
    """
    Parse a decode_meta JSON and return a dict with keys:
        forward          – {(table, col): {str → int}}
        scale_factors    – {(table, col): int}
        attr_compare_precomputed – {(lc, op, rc): (table, col, val)}
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    out_forward: Dict[Tuple[str, str], Dict[str, int]] = {}
    out_scale:   Dict[Tuple[str, str], int] = {}
    out_char_code: Set[Tuple[str, str]] = set()
    out_float_decimal: Set[Tuple[str, str]] = set()
    out_meta: Dict[Tuple[str, str], Dict[str, Any]] = {}
    out_deferred: Dict[Tuple[str, str], Dict[str, Any]] = {}

    for table, cols in payload.get("tables", {}).items():
        tname = _norm(table)
        if not isinstance(cols, dict):
            continue
        for col, meta in cols.items():
            cname = _norm(col)
            if not isinstance(meta, dict):
                continue

            out_meta[(tname, cname)] = dict(meta)

            kind = meta.get("kind", "")

            # CHAR(1) column: .bin stores ASCII ordinals, _enc.bin stores compact rank
            if kind == "char_code":
                out_char_code.add((tname, cname))

            # Compact PREFIX#NUMBER encoding (e.g. MFGR#1, Brand#11, Manufacturer#1)
            if kind == "prefix_hash_int":
                pfx = str(meta.get("prefix", ""))
                phi_nums = meta.get("nums", [])
                phi_fwd: Dict[str, int] = {pfx + str(n): i for i, n in enumerate(phi_nums)}
                if phi_fwd:
                    out_forward[(tname, cname)] = phi_fwd
                    src = meta.get("source_column")
                    if isinstance(src, str) and src.strip():
                        src_key = (tname, _norm(src))
                        if src_key not in out_forward:
                            out_forward[src_key] = dict(phi_fwd)

            # Compact PREFIX#ZERO_PADDED_N encoding (e.g. Customer#000000001)
            elif kind == "prefix_hash_padded_int":
                pfx = str(meta.get("prefix", ""))
                pad = int(meta.get("pad", 1))
                if "nums" in meta:
                    padded_fwd: Dict[str, int] = {
                        pfx + str(n).zfill(pad): i for i, n in enumerate(meta["nums"])
                    }
                else:
                    min_n = int(meta.get("min", 0))
                    max_n = int(meta.get("max", 0))
                    padded_fwd = {
                        pfx + str(n).zfill(pad): i
                        for i, n in enumerate(range(min_n, max_n + 1))
                    }
                if padded_fwd:
                    out_forward[(tname, cname)] = padded_fwd
                    src = meta.get("source_column")
                    if isinstance(src, str) and src.strip():
                        src_key = (tname, _norm(src))
                        if src_key not in out_forward:
                            out_forward[src_key] = dict(padded_fwd)

            # Dict-encoded column – build forward map from reverse_map
            reverse = meta.get("reverse_map")
            if isinstance(reverse, dict):
                fwd: Dict[str, int] = {}
                for enc_str, raw in reverse.items():
                    try:
                        fwd[str(raw)] = int(enc_str)
                    except Exception:
                        continue
                if fwd:
                    out_forward[(tname, cname)] = fwd
                    # Also register under source_column alias if present
                    src = meta.get("source_column")
                    if isinstance(src, str) and src.strip():
                        src_key = (tname, _norm(src))
                        if src_key not in out_forward:
                            out_forward[src_key] = dict(fwd)

            if kind == "dict" and (tname, cname) not in out_forward:
                out_deferred[(tname, cname)] = dict(meta)

            # Float DECIMAL column (IEEE-754 float32 storage)
            if kind == "float_decimal":
                out_float_decimal.add((tname, cname))

            # Scaled DECIMAL column
            if kind == "scaled_decimal_2":
                scale = int(meta.get("scale", 2))
                out_scale[(tname, cname)] = 10 ** scale

    # Attr-compare-precomputed section
    out_acp: Dict[Tuple[str, str, str], Tuple[str, str, int]] = {}
    for entry in payload.get("attr_compare_precomputed", []):
        try:
            lc, op, rc, tbl, pcol, val = entry
            sql_op = _IR_TYPE_TO_SQL_OP.get(op, op)
            out_acp[(_norm(lc), sql_op, _norm(rc))] = (_norm(tbl), _norm(pcol), int(val))
        except Exception:
            continue

    return {
        "forward": out_forward,
        "scale_factors": out_scale,
        "attr_compare_precomputed": out_acp,
        "char_code_columns": out_char_code,
        "float_decimal_columns": out_float_decimal,
        "column_meta": out_meta,
        "deferred_forward_meta": out_deferred,
    }


# ---------------------------------------------------------------------------
# Candidate path resolution
# ---------------------------------------------------------------------------

def _candidate_meta_paths(repo_root: Path) -> list[Path]:
    paths: list[Path] = []
    seen: set[str] = set()

    def _add(p: Path) -> None:
        k = str(p)
        if k not in seen:
            seen.add(k)
            paths.append(p)

    env_paths = os.environ.get("FASCALSQL_DECODE_META_PATHS", "")
    if env_paths.strip():
        for token in env_paths.split(os.pathsep):
            t = token.strip()
            if t:
                _add(Path(t).expanduser())

    # Explicit runtime data directories override all repo defaults.
    ssb_dir = os.environ.get("FASCALSQL_SSB_DATA_DIR", "").strip()
    if ssb_dir:
        _add(Path(ssb_dir) / "ssb_decode_meta.json")
    tpch_dir = os.environ.get("FASCALSQL_TPCH_DATA_DIR", "").strip()
    if tpch_dir:
        _add(Path(tpch_dir) / "tpch_decode_meta.json")

    default_data_dir = os.environ.get("FASCALSQL_DATA_DIR", "").strip()
    if default_data_dir:
        _add(Path(default_data_dir) / "ssb_decode_meta.json")
        _add(Path(default_data_dir) / "tpch_decode_meta.json")

    # Runtime data directory paths generated by prepare_bin scripts.
    _add(repo_root / "tmp" / "fascalsql_ssb_data" / "ssb_decode_meta.json")
    _add(repo_root / "tmp" / "fascalsql_tpch_data" / "tpch_decode_meta.json")

    # Repo-side decode_meta files (committed snapshot, used as fallback when runtime data
    # is not yet available, e.g. in CI or a fresh checkout before running prepare scripts)
    _add(repo_root / "fascalsql" / "configs" / "tpch_decode_meta.json")
    _add(repo_root / "fascalsql" / "configs" / "ssb_decode_meta.json")
    _add(repo_root / "fascalsql" / "configs" / "verification" / "tpch_decode_meta.json")
    _add(repo_root / "fascalsql" / "configs" / "verification" / "ssb_decode_meta.json")

    # Legacy path fallback (old default was /tmp/fascalsql_tpch_sf1)
    _add(repo_root / "tmp" / "fascalsql_tpch_sf1" / "tpch_decode_meta.json")

    return paths


# ---------------------------------------------------------------------------
# Public loader (singleton)
# ---------------------------------------------------------------------------

_ENCODING_CATALOG: Optional[EncodingCatalog] = None


def load_encoding_catalog() -> EncodingCatalog:
    global _ENCODING_CATALOG
    if _ENCODING_CATALOG is not None:
        return _ENCODING_CATALOG

    repo_root = Path(__file__).resolve().parents[3]

    merged_forward: Dict[Tuple[str, str], Dict[str, int]] = {}
    merged_scale:   Dict[Tuple[str, str], int] = {}
    merged_acp:     Dict[Tuple[str, str, str], Tuple[str, str, int]] = {}
    merged_char_code: Set[Tuple[str, str]] = set()
    merged_float_decimal: Set[Tuple[str, str]] = set()
    merged_meta: Dict[Tuple[str, str], Dict[str, Any]] = {}
    merged_deferred: Dict[Tuple[str, str], Dict[str, Any]] = {}

    for p in _candidate_meta_paths(repo_root):
        if p.exists():
            data = _load_one_meta(p)
            for k, v in data["forward"].items():
                # Skip forward maps from lower-priority meta files when a
                # higher-priority file already registered the same column as
                # deferred.  The deferred entry will lazy-load from the
                # correct family's data directory at lookup time.
                if k in merged_deferred:
                    continue
                merged_forward.setdefault(k, v)
            for k, v in data["scale_factors"].items():
                merged_scale.setdefault(k, v)
            for k, v in data["attr_compare_precomputed"].items():
                merged_acp.setdefault(k, v)
            merged_char_code.update(data.get("char_code_columns", set()))
            # Only add float_decimal columns that haven't been overridden by a
            # higher-priority meta file with a non-float kind.
            for fd_col in data.get("float_decimal_columns", set()):
                existing_meta = merged_meta.get(fd_col)
                if existing_meta is not None and existing_meta.get("kind") != "float_decimal":
                    continue  # higher-priority meta says this column is NOT float
                merged_float_decimal.add(fd_col)
            for k, v in data.get("column_meta", {}).items():
                merged_meta.setdefault(k, v)
            for k, v in data.get("deferred_forward_meta", {}).items():
                merged_deferred.setdefault(k, v)

    _ENCODING_CATALOG = EncodingCatalog(
        forward=merged_forward,
        deferred_forward_meta=merged_deferred,
        scale_factors=merged_scale,
        attr_compare_precomputed=merged_acp,
        char_code_columns=merged_char_code,
        float_decimal_columns=merged_float_decimal,
        column_meta=merged_meta,
    )
    return _ENCODING_CATALOG


def reset_encoding_catalog() -> None:
    """Force reload on next call to load_encoding_catalog() (useful for tests)."""
    global _ENCODING_CATALOG
    _ENCODING_CATALOG = None
