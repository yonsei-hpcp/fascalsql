#!/usr/bin/env python3
"""
TPC-H resident FaScalSQL 결과 검증: 기대 결과(golden)와 실행 결과 비교.

- **22개 쿼리 모두** 동일하게 전체 결과(테이블) 기준으로 비교. Q16만이 아닌 tpch_q1~tpch_q22 전부.
- FaScalSQL stdout: ROW: a|b|c 또는 Result: / group_key= sum= 파싱 → 행 리스트로 변환 후 행/열 단위 비교.
- 권장 Golden: 동일 SQL(`docs/schemas/tpch/q*.sql`)을 DuckDB로 실행해 생성한 lines 형식
  (`build_tpch_golden_from_docs_sql.py`).
- --save-golden: 현재 실행 결과를 lines 형식으로 저장 (임시/디버그용).

사용 예:
  python build_tpch_golden_from_docs_sql.py --output fascalsql/configs/tpch_golden.json
  python verify_tpch_results.py --data-dir /path/to/tpch_bin --golden fascalsql/configs/tpch_golden.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    from fascalsql.python.utils.verification_core import (
        build_pred_flags,
        build_query_pred_flags,
        compare_rows_with_golden,
        load_duckdb_reference_entry,
        parse_predicate_id_list,
        resolve_queries,
        run_standalone_query,
        sort_rows_for_compare,
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
        sort_rows_for_compare,
        validate_query_result_schema,
    )

# TPC-H 쿼리 목록 (tpch_queries.ALL_TPCH_QUERIES와 동기화)
TPCH_QUERIES = [
    f"tpch_q{i}" for i in range(1, 23)
]

def parse_fascalsql_stdout(stdout: str) -> Tuple[Optional[float], List[List[str]]]:
    """
    FaScalSQL stdout에서 (Kernel ms, result rows) 추출.
    - ROW: a|b|c → [[a,b,c], ...] (전체 테이블)
    - Result: X → [[X]]
    - group_key= K sum= V → [[K, V], ...]
    """
    kernel_ms: Optional[float] = None
    result_rows: List[List[str]] = []
    legacy_lines: List[Tuple[str, str]] = []
    scalar_result: Optional[str] = None

    for raw_line in stdout.splitlines():
        line = raw_line.rstrip("\r\n")
        stripped = line.strip()
        m = re.match(r"Kernel:\s*([\d.]+)\s*ms", stripped)
        if m:
            kernel_ms = float(m.group(1))
            continue
        m = re.match(r"ROW:\s*(.*)", line)
        if m:
            row = m.group(1).split("|")
            result_rows.append(row)
            continue
        m = re.match(r"group_key=(\S+)\s+sum=(\S+)", stripped)
        if m:
            legacy_lines.append((m.group(1), m.group(2)))
            continue
        m = re.match(r"Result:\s*(\S+)", stripped)
        if m:
            scalar_result = m.group(1)
            continue

    if not result_rows:
        if scalar_result is not None and not legacy_lines:
            result_rows = [[scalar_result]]
        else:
            result_rows = [[k, v] for k, v in legacy_lines]
    return kernel_ms, result_rows


def normalize_result_rows(rows: List[List[str]]) -> List[List[str]]:
    """비교용: 정렬된 행 리스트 (전체 테이블 비교 시). 2열 이하는 (k,v) 정렬."""
    return sort_rows_for_compare(rows)


@lru_cache(maxsize=None)
def _load_runtime_dict(data_dir_str: str, column: str) -> Dict[str, str]:
    dict_path = Path(data_dir_str) / f"{column}_dict.json"
    if not dict_path.exists():
        return {}
    try:
        with dict_path.open("r", encoding="utf-8") as f:
            forward_map = json.load(f)
    except Exception:
        return {}
    if not isinstance(forward_map, dict):
        return {}
    return {str(v): str(k) for k, v in forward_map.items()}


def decode_cell(
    value: str,
    col_meta: Optional[Dict[str, Any]],
    *,
    data_dir: Optional[Path] = None,
    column: Optional[str] = None,
) -> str:
    """Decode one encoded cell value using metadata."""
    if not col_meta:
        return value
    kind = str(col_meta.get("kind", "")).strip()
    if kind == "dict":
        reverse_map = {}
        if data_dir is not None and column:
            reverse_map = _load_runtime_dict(str(data_dir), column)
        if not reverse_map:
            reverse_map = col_meta.get("reverse_map", {})
        return str(reverse_map.get(str(value), value))
    if kind == "prefix_hash_int":
        try:
            nums = list(col_meta.get("nums", []))
            idx = int(value)
            if 0 <= idx < len(nums):
                return f"{col_meta.get('prefix', '')}{nums[idx]}"
        except Exception:
            return value
        return value
    if kind == "prefix_hash_padded_int":
        try:
            idx = int(value)
            nums = col_meta.get("nums")
            if isinstance(nums, list):
                if 0 <= idx < len(nums):
                    num = int(nums[idx])
                else:
                    return value
            else:
                min_n = int(col_meta.get("min", 0))
                max_n = int(col_meta.get("max", -1))
                num = min_n + idx
                if num > max_n:
                    return value
            prefix = str(col_meta.get("prefix", ""))
            pad = int(col_meta.get("pad", 0))
            return f"{prefix}{num:0{pad}d}"
        except Exception:
            return value
    if kind == "char_code":
        try:
            return chr(int(value))
        except Exception:
            return value
    if kind == "int":
        reverse_map = col_meta.get("reverse_map", {})
        return str(reverse_map.get(str(value), value))
    if kind == "date":
        try:
            v = int(value)
            yyyy = v // 10000
            mm = (v // 100) % 100
            dd = v % 100
            return f"{yyyy:04d}-{mm:02d}-{dd:02d}"
        except Exception:
            return value
    if kind == "float_decimal":
        try:
            return f"{float(value):.4f}"
        except Exception:
            return value
    if kind == "scaled_decimal_2" or ("scale" in col_meta):
        try:
            scale = int(col_meta.get("scale", 2))
            iv = int(value)
            sign = "-" if iv < 0 else ""
            av = abs(iv)
            base = 10 ** max(scale, 0)
            whole = av // base
            frac = av % base
            if scale > 0:
                return f"{sign}{whole}.{frac:0{scale}d}"
            return f"{sign}{whole}"
        except Exception:
            return value
    return value


def decode_rows_for_query(
    query: str,
    rows: List[List[str]],
    decode_meta: Dict[str, Any],
    result_schema: Dict[str, Any],
    data_dir: Optional[Path] = None,
) -> List[List[str]]:
    """
    Decode query result rows by schema.
    schema format:
      {
        "tpch_q4": {
          "columns": [
            {"table":"orders","column":"o_orderpriority_enc"},
            {"type":"int"}
          ]
        }
      }
    """
    qschema = result_schema.get(query, {})
    decoded_rows: List[List[str]] = [[str(cell) for cell in row] for row in rows]

    postprocess = str(qschema.get("postprocess", "")).strip()
    if postprocess == "q14_promo_ratio" and decoded_rows:
        try:
            promo = float(decoded_rows[0][0])
            total = float(decoded_rows[0][1])
            ratio = 0.0 if total == 0 else (100.0 * promo / total)
            return [[f"{ratio:.6f}"]]
        except Exception:
            return decoded_rows
    elif postprocess == "q8_mkt_share" and decoded_rows:
        out_rows = []
        for row in decoded_rows:
            try:
                year_val = int(row[0])
                year = str(year_val if year_val >= 1992 else (year_val + 1992))
                num = float(row[1])
                denom = float(row[2])
                ratio = 0.0 if denom == 0 else (num / denom)
                out_rows.append([year, f"{ratio:.16f}"])
            except Exception:
                out_rows.append(row)
        return out_rows
    elif postprocess == "q17_avg" and decoded_rows:
        # Q17 SQL: SUM(l_extendedprice) / 7.0 as avg_yearly
        # The binary outputs raw SUM; divide by 7.0 here
        out_rows = []
        for row in decoded_rows:
            try:
                val = float(row[0]) / 7.0
                out_rows.append([f"{val:.10f}"])
            except Exception:
                out_rows.append(row)
        return out_rows

    columns = qschema.get("columns", [])
    if not columns or not decoded_rows:
        return decoded_rows

    out_rows: List[List[str]] = []
    for row in decoded_rows:
        out_row: List[str] = []
        for idx, cell in enumerate(row):
            spec = columns[idx] if idx < len(columns) and isinstance(columns[idx], dict) else {}
            table = str(spec.get("table", "")).strip()
            column = str(spec.get("column", "")).strip()
            decode_table = str(spec.get("decode_table", "")).strip()
            decode_column = str(spec.get("decode_column", "")).strip()
            spec_type = str(spec.get("type", "")).strip()
            if decode_table and decode_column:
                col_meta = decode_meta.get("tables", {}).get(decode_table, {}).get(decode_column)
                decoded = decode_cell(cell, col_meta, data_dir=data_dir, column=decode_column)
                if "format" in spec:
                    try:
                        decoded = str(spec.get("format", "%s")) % float(decoded)
                    except Exception:
                        pass
                out_row.append(decoded)
                continue
            if table and column:
                col_meta = decode_meta.get("tables", {}).get(table, {}).get(column)
                decoded = decode_cell(cell, col_meta, data_dir=data_dir, column=column)
                if "format" in spec:
                    try:
                        decoded = str(spec.get("format", "%s")) % float(decoded)
                    except Exception:
                        pass
                out_row.append(decoded)
                continue
            if spec_type == "float_decimal":
                try:
                    fmt = str(spec.get("format", "%.4f"))
                    out_row.append(fmt % float(cell))
                except Exception:
                    out_row.append(str(cell))
                continue
            if spec_type == "year_enc":
                try:
                    base = int(spec.get("base", 0))
                    year = int(cell)
                    out_row.append(str(base + year) if base and year < base else str(year))
                except Exception:
                    out_row.append(str(cell))
                continue
            if spec_type == "float_avg":
                try:
                    fmt = str(spec.get("format", "%.6f"))
                    out_row.append(fmt % float(cell))
                except Exception:
                    out_row.append(str(cell))
                continue
            out_row.append(str(cell))
        out_rows.append(out_row)
    return out_rows


def get_default_pred_flags(query_name: str) -> List[int]:
    """Plan/codegen payload에서 predicate 개수를 읽어 기본 GPU 배치를 반환한다."""
    return build_query_pred_flags(query_name, placement_mode="gpu_all")


def run_fascalsql(
    query: str,
    data_dir: Path,
    pred_flags: List[int],
    timeout_s: int = 300,
    *,
    report_latency: bool = False,
) -> Tuple[int, str, float, Optional[Dict[str, Any]]]:
    """Standalone binary로 FaScalSQL query 실행. 반환: (returncode, stdout, wall_seconds, timing_dict or None)."""
    returncode, output = run_standalone_query(
        query,
        pred_flags=pred_flags,
        timeout_s=timeout_s,
        data_dir=data_dir,
    )
    return returncode, output, 0.0, None


def compare_with_golden(
    current_rows: List[List[str]],
    golden: Dict[str, Any],
) -> Tuple[bool, str]:
    return compare_rows_with_golden(current_rows, golden)


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify TPC-H FaScalSQL results against golden.")
    parser.add_argument("--data-dir", type=str, required=True, help="FaScalSQL .bin data directory")
    parser.add_argument("--golden", type=str, default=None, help="Golden JSON path (default: configs/tpch_golden.json)")
    parser.add_argument("--duckdb-db", type=str, default=None, help="DuckDB reference DB to execute canonical SQL against")
    parser.add_argument(
        "--decode-meta",
        type=str,
        default=None,
        help="Decode metadata JSON path (default: <data-dir>/tpch_decode_meta.json, fallback: fascalsql/configs/tpch_decode_meta.json)",
    )
    parser.add_argument("--result-schema", type=str, default=None, help="Result schema JSON path (default: configs/tpch_result_schema.json)")
    parser.add_argument("--require-golden", action="store_true", help="Fail if golden file does not exist")
    parser.add_argument("--save-golden", action="store_true", help="Save current run results as golden")
    parser.add_argument("--timeout", type=int, default=300, help="Per-query timeout (seconds)")
    parser.add_argument("--repo-root", type=str, default=None, help="Repo root (default: script parent parent)")
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
    parser.add_argument(
        "--report-latency",
        action="store_true",
        help="Report per-query execution latency breakdown (contract_gen_ms, runner_total_ms, kernel_ms)",
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root) if args.repo_root else Path(__file__).resolve().parent.parent.parent.parent
    data_dir = Path(args.data_dir)
    if not data_dir.is_dir():
        print(f"Error: data-dir not a directory: {data_dir}", file=sys.stderr)
        return 1

    config_dir = repo_root / "fascalsql" / "configs" / "verification"
    golden_path = Path(args.golden) if args.golden else config_dir / "tpch_golden.json"
    if args.decode_meta:
        decode_meta_path = Path(args.decode_meta)
    else:
        runtime_decode_meta = data_dir / "tpch_decode_meta.json"
        decode_meta_path = runtime_decode_meta if runtime_decode_meta.exists() else config_dir / "tpch_decode_meta.json"
    schema_path = Path(args.result_schema) if args.result_schema else config_dir / "tpch_result_schema.json"
    duckdb_db = Path(args.duckdb_db) if args.duckdb_db else None
    golden_data: Dict[str, Dict[str, Any]] = {}
    decode_meta: Dict[str, Any] = {}
    result_schema: Dict[str, Any] = {}
    if duckdb_db is not None and not duckdb_db.exists():
        print(f"Error: DuckDB reference DB not found: {duckdb_db}", file=sys.stderr)
        return 1
    if duckdb_db is None and golden_path.exists() and not args.save_golden:
        with open(golden_path, "r", encoding="utf-8") as f:
            golden_data = json.load(f)
    elif duckdb_db is None and args.require_golden and not args.save_golden:
        print(f"Error: golden file not found: {golden_path}", file=sys.stderr)
        return 1
    if decode_meta_path.exists():
        with open(decode_meta_path, "r", encoding="utf-8") as f:
            decode_meta = json.load(f)
    if schema_path.exists():
        with open(schema_path, "r", encoding="utf-8") as f:
            result_schema = json.load(f)

    passed = 0
    failed: List[Tuple[str, str]] = []
    results_for_save: Dict[str, Dict[str, Any]] = {}
    latency_records: List[Tuple[str, Dict[str, Any]]] = []
    try:
        selected_queries = resolve_queries(args.queries, TPCH_QUERIES)
        selected_cpu_ids = parse_predicate_id_list(args.cpu_predicate_ids)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    for q in selected_queries:
        try:
            pred_flags = build_query_pred_flags(
                q,
                placement_mode=args.pred_mode,
                cpu_predicate_ids=selected_cpu_ids,
            )
        except ValueError as exc:
            failed.append((q, str(exc)))
            continue
        rc, out, _, timing = run_fascalsql(
            q, data_dir, pred_flags, args.timeout, report_latency=args.report_latency
        )
        if timing is not None:
            latency_records.append((q, timing))

        if rc != 0:
            failed.append((q, "run failed"))
            if args.save_golden:
                results_for_save[q] = {"error": out[:200]}
            continue

        kernel_ms, result_rows = parse_fascalsql_stdout(out)
        decoded_rows = decode_rows_for_query(q, result_rows, decode_meta, result_schema, data_dir=data_dir)
        norm = normalize_result_rows(decoded_rows)

        if args.save_golden:
            # 모든 TPC-H 쿼리는 전체 테이블(lines) 형식으로 저장
            results_for_save[q] = {"lines": [list(r) for r in norm]}
            passed += 1
            continue

        if duckdb_db is None:
            if q not in golden_data:
                failed.append((q, "no golden"))
                continue
            golden_entry = golden_data[q]
        else:
            try:
                golden_entry = load_duckdb_reference_entry(q, duckdb_db)
            except Exception as exc:
                failed.append((q, f"DuckDB reference error: {exc}"))
                continue

        schema_ok, schema_msg = validate_query_result_schema(
            benchmark="tpch",
            query=q,
            rows=norm,
            result_schema=result_schema,
            decode_meta=decode_meta,
            repo_root=repo_root,
            golden_entry=golden_entry,
        )
        if not schema_ok:
            failed.append((q, schema_msg))
            continue

        ok, msg = compare_with_golden(norm, golden_entry)
        if ok:
            passed += 1
        else:
            failed.append((q, msg))

    if args.save_golden:
        config_dir.mkdir(parents=True, exist_ok=True)
        with open(golden_path, "w", encoding="utf-8") as f:
            json.dump(results_for_save, f, indent=2, ensure_ascii=False)
        print(f"Saved golden for {len(results_for_save)} queries to {golden_path}")
        return 0 if not any("error" in v for v in results_for_save.values()) else 1

    total = len(selected_queries)
    print(f"TPC-H: {passed}/{total} passed")
    for q, msg in failed:
        print(f"  FAIL {q}: {msg}")
    if args.report_latency and latency_records:
        print("")
        print("Latency breakdown (ms):")
        for q, t in latency_records:
            k = t.get("kernel_ms")
            k_str = f"{k:.2f}" if k is not None else "N/A"
            print(f"  {q}: contract_gen={t.get('contract_gen_ms', 0):.1f} runner_total={t.get('runner_total_ms', 0):.1f} kernel={k_str}")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
