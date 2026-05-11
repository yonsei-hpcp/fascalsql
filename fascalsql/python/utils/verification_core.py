from __future__ import annotations

import json
import os
import re
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    import duckdb  # type: ignore
except Exception:
    duckdb = None


PREDICATE_PLACEMENT_MODES = (
    "gpu_all",
    "cpu_all",
    "default_cpu",
    "alternating",
    "selected_cpu",
)

_CREATE_TABLE_RE = re.compile(
    r"CREATE\s+TABLE\s+([A-Za-z_]\w*)\s*\((.*?)\)\s*;",
    re.IGNORECASE | re.DOTALL,
)
_COLUMN_RE = re.compile(
    r"^\s*([A-Za-z_]\w*)\s+([A-Za-z]+(?:\([^)]*\))?)",
    re.IGNORECASE,
)
_CONSTRAINT_KEYWORDS = {"primary", "foreign", "constraint", "unique", "check"}
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_IDENT_RE = re.compile(r"^[A-Za-z_]\w*$")

_REPO_ROOT = Path(__file__).resolve().parents[3]

try:
    from fascalsql.python.utils.encoding_catalog import reset_encoding_catalog
    from fascalsql.python.utils.sql_query_registry import load_docs_sql_queries
except ImportError:
    from .encoding_catalog import reset_encoding_catalog  # type: ignore
    from .sql_query_registry import load_docs_sql_queries  # type: ignore


def _strip_comments(text: str) -> str:
    return "\n".join(line.split("--", 1)[0] for line in text.splitlines())


def _split_top_level(block: str) -> List[str]:
    items: List[str] = []
    cur: List[str] = []
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
    alias_map: Dict[str, str] = {}
    select_list = sql[select_pos + len("SELECT"):from_pos]
    for item in _split_top_level(select_list):
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
    group_items = _split_top_level(sql[clause_start:clause_end])
    rewritten_items: List[str] = []
    changed = False
    for item in group_items:
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


def _parse_ddl_types(ddl_text: str) -> Dict[str, Dict[str, str]]:
    tables: Dict[str, Dict[str, str]] = {}
    for match in _CREATE_TABLE_RE.finditer(_strip_comments(ddl_text)):
        table_name = match.group(1).strip().lower()
        cols: Dict[str, str] = {}
        for item in _split_top_level(match.group(2)):
            item = item.strip()
            if not item:
                continue
            tokens = item.split()
            if tokens and tokens[0].lower() in _CONSTRAINT_KEYWORDS:
                continue
            column_match = _COLUMN_RE.match(item)
            if column_match:
                cols[column_match.group(1).strip().lower()] = column_match.group(2).strip().upper()
        tables[table_name] = cols
    return tables


@lru_cache(maxsize=None)
def _load_ddl_types(repo_root_str: str, benchmark: str) -> Dict[str, Dict[str, str]]:
    ddl_path = Path(repo_root_str) / "docs" / "schemas" / f"{benchmark}.ddl"
    if not ddl_path.exists():
        return {}
    return _parse_ddl_types(ddl_path.read_text(encoding="utf-8"))


def load_ablation_predicates(ablation_path: Path) -> List[Dict[str, Any]]:
    if not ablation_path.exists():
        return []
    with ablation_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    predicates = payload.get("predicates", [])
    return predicates if isinstance(predicates, list) else []


def build_pred_flags_from_payload(
    payload: Dict[str, Any],
    placement_mode: str = "gpu_all",
    cpu_predicate_ids: Optional[Sequence[int]] = None,
) -> List[int]:
    preds = payload.get("predicates", [])
    if not isinstance(preds, list) or not preds:
        return []
    if placement_mode == "gpu_all":
        return [0 for _ in preds]
    if placement_mode == "cpu_all":
        return [1 for _ in preds]
    if placement_mode == "default_cpu":
        return [1 if bool(pred.get("default_run_on_cpu", False)) else 0 for pred in preds]
    if placement_mode == "alternating":
        return [idx % 2 for idx, _ in enumerate(preds)]
    if placement_mode == "selected_cpu":
        cpu_ids = {int(pid) for pid in (cpu_predicate_ids or [])}
        invalid = sorted(pid for pid in cpu_ids if pid < 0 or pid >= len(preds))
        if invalid:
            raise ValueError(f"Predicate ids out of range: {invalid}")
        return [1 if idx in cpu_ids else 0 for idx, _ in enumerate(preds)]
    raise ValueError(f"Unsupported predicate placement mode: {placement_mode}")


def build_pred_flags(
    ablation_path: Path,
    placement_mode: str = "gpu_all",
    cpu_predicate_ids: Optional[Sequence[int]] = None,
) -> List[int]:
    return build_pred_flags_from_payload(
        {"predicates": load_ablation_predicates(ablation_path)},
        placement_mode=placement_mode,
        cpu_predicate_ids=cpu_predicate_ids,
    )


def _load_ablation_payload_from_file(query_name: str) -> Dict[str, Any]:
    """Load predicate list from ablation JSON config file."""
    for family in ("tpch", "ssb"):
        ablation_path = _REPO_ROOT / "fascalsql" / "configs" / "ablation" / f"{query_name}_ablation.json"
        if ablation_path.exists():
            return {"predicates": load_ablation_predicates(ablation_path)}
    return {"predicates": []}


def build_query_pred_flags(
    query_name: str,
    placement_mode: str = "gpu_all",
    cpu_predicate_ids: Optional[Sequence[int]] = None,
) -> List[int]:
    return build_pred_flags_from_payload(
        _load_ablation_payload_from_file(query_name),
        placement_mode=placement_mode,
        cpu_predicate_ids=cpu_predicate_ids,
    )


def parse_predicate_id_list(raw: Optional[str]) -> List[int]:
    if raw is None or not str(raw).strip():
        return []
    out: List[int] = []
    for token in str(raw).split(","):
        token = token.strip()
        if token:
            out.append(int(token))
    return out


def resolve_queries(requested: Optional[Sequence[str]], all_queries: Sequence[str]) -> List[str]:
    if not requested:
        return list(all_queries)
    selected = [str(query).strip() for query in requested if str(query).strip()]
    unknown = sorted({query for query in selected if query not in all_queries})
    if unknown:
        raise ValueError(f"Unknown queries: {', '.join(unknown)}")
    return selected


def _format_reference_value(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def normalize_sql_for_duckdb(sql: str) -> str:
    def _replace_date_with_modifier(match: re.Match[str]) -> str:
        base = match.group(1)
        sign = match.group(2)
        amount = int(match.group(3))
        unit = match.group(4).upper()
        if sign == "-":
            amount = -amount
        unit = unit.rstrip("S")
        return f"(DATE '{base}' + INTERVAL {amount} {unit})"

    sql = re.sub(
        r"(?i)\bdate\s*\(\s*'([^']+)'\s*,\s*'([+-])\s*(\d+)\s*(day|days|month|months|year|years)'\s*\)",
        _replace_date_with_modifier,
        sql,
    )
    sql = re.sub(
        r"(?i)\bdate\s*\(\s*'([^']+)'\s*\)",
        lambda match: f"DATE '{match.group(1)}'",
        sql,
    )
    # Remove ::fixeddecimal casts (not supported by DuckDB)
    sql = re.sub(r"'(\d+(?:\.\d+)?)'::fixeddecimal", r"\1", sql)
    sql = re.sub(r"(\d+(?:\.\d+)?)::fixeddecimal", r"\1", sql)
    sql = _expand_group_by_aliases_for_duckdb(sql)
    return sql


def load_duckdb_reference_entry(
    query_name: str,
    duckdb_db: Path,
    *,
    sql_text: Optional[str] = None,
) -> Dict[str, Any]:
    if duckdb is None:
        raise RuntimeError("duckdb package is not installed")
    if not duckdb_db.exists():
        raise FileNotFoundError(f"DuckDB reference database not found: {duckdb_db}")

    query_sql = sql_text
    if query_sql is None:
        query_sql = load_docs_sql_queries().get(query_name)
    if not query_sql:
        raise KeyError(f"No SQL registered for {query_name}")

    conn = duckdb.connect(str(duckdb_db), read_only=True)
    try:
        cursor = conn.execute(normalize_sql_for_duckdb(query_sql))
        rows = cursor.fetchall()
        schema = [
            {
                "name": str(desc[0]) if desc else "",
                "duckdb_type": str(desc[1]) if len(desc) > 1 else "",
            }
            for desc in (cursor.description or [])
        ]
        return {
            "lines": [[_format_reference_value(cell) for cell in row] for row in rows],
            "schema": schema,
        }
    finally:
        conn.close()


def run_standalone_query(
    query_name: str,
    *,
    pred_flags: Optional[Sequence[int]] = None,
    timeout_s: int = 300,
    data_dir: Optional[Path] = None,
) -> Tuple[int, str]:
    """Run a standalone query binary and return (returncode, output)."""
    import time
    if data_dir is None:
        return -1, f"Query execution requires data_dir for {query_name}"

    build_dir = _REPO_ROOT / "fascalsql" / "host" / "build" / "standalone"
    binary = build_dir / query_name
    if not binary.exists():
        return -1, f"Standalone binary not found: {binary}"

    resolved_data_dir = str(Path(data_dir).resolve())
    cmd = [str(binary), "0", resolved_data_dir] + [str(int(flag)) for flag in (pred_flags or [])]

    env = os.environ.copy()
    env["FASCALSQL_DATA_DIR"] = resolved_data_dir

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            env=env,
        )
    except subprocess.TimeoutExpired:
        return -1, f"Timeout ({timeout_s}s)"
    return proc.returncode, proc.stdout + proc.stderr


def _sort_key(row: Sequence[str]) -> Tuple[Tuple[int, float, str], ...]:
    def cell_key(value: str) -> Tuple[int, float, str]:
        stripped = str(value).strip()
        try:
            return (0, float(stripped), stripped)
        except ValueError:
            return (1, 0.0, stripped)

    return tuple(cell_key(value) for value in row)


def sort_rows_for_compare(rows: Sequence[Sequence[str]]) -> List[List[str]]:
    normalized = [[str(cell) for cell in row] for row in rows]
    return sorted(normalized, key=_sort_key)


def compare_rows_with_golden(
    current_rows: Sequence[Sequence[str]],
    golden: Dict[str, Any],
    *,
    tolerance: float = 1e-6,
) -> Tuple[bool, str]:
    rows = [[str(cell) for cell in row] for row in current_rows]
    if "scalar" in golden:
        if len(rows) != 1 or len(rows[0]) != 1:
            return False, f"Expected scalar result, got {len(rows)} rows"
        want = str(golden["scalar"])
        got = str(rows[0][0])
        if got != want:
            try:
                if abs(float(got) - float(want)) > tolerance * (abs(float(want)) + 1.0):
                    return False, f"Scalar mismatch: got {got}, expected {want}"
            except (TypeError, ValueError):
                return False, f"Scalar mismatch: got {got}, expected {want}"
        return True, "OK"

    if "lines" not in golden:
        return False, "Golden has neither 'scalar' nor 'lines'"

    want_rows = [[str(cell) for cell in row] for row in golden["lines"]]
    curr_sorted = sort_rows_for_compare(rows)
    want_sorted = sort_rows_for_compare(want_rows)
    if len(curr_sorted) != len(want_sorted):
        return False, f"Row count mismatch: {len(curr_sorted)} vs {len(want_sorted)}"

    for row_idx, (current_row, want_row) in enumerate(zip(curr_sorted, want_sorted)):
        if len(current_row) != len(want_row):
            return False, f"Row {row_idx}: column count mismatch: {len(current_row)} vs {len(want_row)}"
        for col_idx, (current_value, want_value) in enumerate(zip(current_row, want_row)):
            if current_value == want_value:
                continue
            try:
                current_num = float(current_value)
                want_num = float(want_value)
            except ValueError:
                return False, f"Row {row_idx}, col {col_idx}: '{current_value}' != '{want_value}'"
            if abs(current_num - want_num) > tolerance * max(abs(want_num), 1.0):
                return False, f"Row {row_idx}, col {col_idx}: {current_value} != {want_value}"
    return True, "OK"


def _normalize_logical_type(raw_type: Optional[str]) -> str:
    text = str(raw_type or "").strip().lower()
    if not text:
        return "unknown"
    if any(token in text for token in ("varchar", "char", "text", "string", "dict", "prefix_hash")):
        return "string"
    if text in {"date", "timestamp"} or "date" in text:
        return "date"
    if any(token in text for token in ("float", "double", "real", "decimal", "numeric")):
        return "decimal"
    if any(token in text for token in ("int", "integer", "bigint", "smallint", "tinyint", "hugeint", "ubigint", "year")):
        return "int"
    return "unknown"


def _logical_type_from_meta_kind(kind: Optional[str]) -> str:
    text = str(kind or "").strip().lower()
    if not text:
        return "unknown"
    if text in {"dict", "char_code", "prefix_hash_int", "prefix_hash_padded_int"}:
        return "string"
    if text in {"date"}:
        return "date"
    if text in {"scaled_decimal", "scaled_decimal_2", "float_decimal"}:
        return "decimal"
    if text in {"int", "yearmonth"}:
        return "int"
    return "unknown"


def _normalize_duckdb_schema(golden_entry: Optional[Dict[str, Any]]) -> List[str]:
    if not isinstance(golden_entry, dict):
        return []
    schema = golden_entry.get("schema")
    if not isinstance(schema, list):
        return []
    logical_types: List[str] = []
    for column in schema:
        if not isinstance(column, dict):
            logical_types.append("unknown")
            continue
        logical_types.append(_normalize_logical_type(column.get("duckdb_type") or column.get("type")))
    return logical_types


def _tpch_postprocess_types(postprocess: str) -> List[str]:
    if postprocess == "q14_promo_ratio":
        return ["decimal"]
    if postprocess == "q8_mkt_share":
        return ["int", "decimal"]
    if postprocess == "q17_avg":
        return ["decimal"]
    return []


def _tpch_expected_types(
    query: str,
    result_schema: Dict[str, Any],
    decode_meta: Dict[str, Any],
    repo_root: Path,
) -> List[str]:
    query_schema = result_schema.get(query, {})
    if not isinstance(query_schema, dict):
        return []

    postprocess = str(query_schema.get("postprocess", "")).strip()
    if postprocess:
        return _tpch_postprocess_types(postprocess)

    ddl_types = _load_ddl_types(str(repo_root), "tpch")
    columns = query_schema.get("columns", [])
    expected_types: List[str] = []
    for spec in columns:
        if not isinstance(spec, dict):
            expected_types.append("unknown")
            continue
        explicit_type = _normalize_logical_type(spec.get("type"))
        if explicit_type != "unknown":
            expected_types.append(explicit_type)
            continue

        decode_table = str(spec.get("decode_table") or spec.get("table") or "").strip().lower()
        decode_column = str(spec.get("decode_column") or spec.get("column") or "").strip().lower()
        if decode_table and decode_column:
            meta = decode_meta.get("tables", {}).get(decode_table, {}).get(decode_column)
            logical_type = _logical_type_from_meta_kind(meta.get("kind") if isinstance(meta, dict) else None)
            if logical_type == "unknown":
                ddl_column = decode_column[:-4] if decode_column.endswith("_enc") else decode_column
                logical_type = _normalize_logical_type(ddl_types.get(decode_table, {}).get(ddl_column))
            if logical_type == "unknown" and "format" in spec:
                logical_type = "decimal"
            expected_types.append(logical_type)
            continue

        if "format" in spec:
            expected_types.append("decimal")
        else:
            expected_types.append("unknown")
    return expected_types


def _ssb_expected_types(query: str, result_schema: Dict[str, Any]) -> List[str]:
    query_schema = result_schema.get(query, {})
    if not isinstance(query_schema, dict):
        return []
    columns = query_schema.get("fascal_columns", [])
    if not isinstance(columns, list):
        return []
    expected_types = [_normalize_logical_type(spec.get("type") if isinstance(spec, dict) else None) for spec in columns]
    golden_order = query_schema.get("golden_column_order", list(range(len(expected_types))))
    if not isinstance(golden_order, list):
        return expected_types
    reordered: List[str] = []
    for idx in golden_order:
        if isinstance(idx, int) and 0 <= idx < len(expected_types):
            reordered.append(expected_types[idx])
    return reordered if reordered else expected_types


def _value_matches_logical_type(value: str, logical_type: str) -> bool:
    if logical_type in {"unknown", "string"}:
        return True
    stripped = str(value).strip()
    if logical_type == "int":
        return bool(re.fullmatch(r"-?\d+", stripped))
    if logical_type == "decimal":
        try:
            float(stripped)
            return True
        except ValueError:
            return False
    if logical_type == "date":
        return bool(_DATE_RE.fullmatch(stripped))
    return True


def validate_query_result_schema(
    *,
    benchmark: str,
    query: str,
    rows: Sequence[Sequence[str]],
    result_schema: Dict[str, Any],
    decode_meta: Optional[Dict[str, Any]] = None,
    repo_root: Optional[Path] = None,
    golden_entry: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, str]:
    if benchmark == "tpch":
        if repo_root is None:
            return False, "Missing repo_root for TPC-H schema validation"
        expected_types = _tpch_expected_types(query, result_schema, decode_meta or {}, repo_root)
    elif benchmark == "ssb":
        expected_types = _ssb_expected_types(query, result_schema)
    else:
        return False, f"Unsupported benchmark for schema validation: {benchmark}"

    golden_types = _normalize_duckdb_schema(golden_entry)
    if golden_types and expected_types and len(golden_types) == len(expected_types):
        mismatches = [
            (idx, expected, golden_type)
            for idx, (expected, golden_type) in enumerate(zip(expected_types, golden_types))
            if expected != "unknown" and golden_type != "unknown" and expected != golden_type
        ]
        if mismatches:
            idx, expected, golden_type = mismatches[0]
            return False, (
                f"Schema mismatch for {query}: column {idx} expected {expected} from verifier schema, "
                f"golden reports {golden_type}"
            )

    if not expected_types:
        expected_types = golden_types
    if not expected_types:
        return True, "OK"

    for row_idx, row in enumerate(rows):
        if len(row) != len(expected_types):
            return False, (
                f"Schema mismatch for {query}: expected {len(expected_types)} columns, got {len(row)} "
                f"on row {row_idx}"
            )
        for col_idx, (value, logical_type) in enumerate(zip(row, expected_types)):
            if not _value_matches_logical_type(str(value), logical_type):
                return False, (
                    f"Type mismatch for {query}: row {row_idx} col {col_idx} value '{value}' "
                    f"is not a valid {logical_type}"
                )
    return True, "OK"
