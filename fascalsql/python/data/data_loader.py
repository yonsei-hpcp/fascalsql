#!/usr/bin/env python3
"""
Optimized streaming data loader with two-pass dictionary encoding.
This standalone script focuses on a two-pass approach for a 6M-row SSB dataset,
targeting batch processing, string dictionary encoding, and binary output with
progress reporting.
"""

import csv
import struct
import sys
import os
import time
import argparse
import json
import multiprocessing as mp
import subprocess
from datetime import datetime


HEADER_MAGIC = b"FST0"
VERSION = 1

# Project-local tmp directory (avoids /tmp which is cleared on reboot)
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_DEFAULT_SSB_DATA_DIR  = os.path.join(_REPO_ROOT, "tmp", "fascalsql_ssb_data")
_DEFAULT_TPCH_DATA_DIR = os.path.join(_REPO_ROOT, "tmp", "fascalsql_tpch_data")
_DEFAULT_SSB_TBL_DIR   = os.path.join(_REPO_ROOT, "tmp", "ssb_tbl")
_DEFAULT_TPCH_TBL_DIR  = os.path.join(_REPO_ROOT, "tmp", "tpch_tbl")


def default_data_dirs() -> dict:
    return {
        "ssb_sf1": os.environ.get("FASCALSQL_SSB_DATA_DIR", _DEFAULT_SSB_DATA_DIR),
        "tpch_sf1": os.environ.get(
            "FASCALSQL_TPCH_DATA_DIR", _DEFAULT_TPCH_DATA_DIR
        ),
    }


def _is_float_safe(s: str) -> bool:
    try:
        float(s)
        return True
    except Exception:
        return False


def _encode_column(args):
    mapping, values = args
    m = mapping
    return [m.get(v, -1) for v in values]


def pass1_build_dicts(input_path, delimiter=",", quotechar='"'):
    """Pass 1: scan input, determine string columns and build dicts for string values."""
    col_names = []
    col_is_string = []
    dicts = []

    total_rows = 0
    with open(input_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=delimiter, quotechar=quotechar)
        header = next(reader, None)
        if header is None:
            raise ValueError("Input file is empty or missing header")
        col_names = header
        ncol = len(col_names)
        col_is_string = [False] * ncol
        dicts = [
            {} for _ in range(ncol)
        ]  # per-column string->int maps (only for strings)

        for row in reader:
            if len(row) < ncol:
                row = row + [""] * (ncol - len(row))
            elif len(row) > ncol:
                row = row[:ncol]
            for j, val in enumerate(row):
                if val is None:
                    val = ""
                if val == "":
                    continue
                if _is_float_safe(val):
                    continue
                if not col_is_string[j]:
                    col_is_string[j] = True
                d = dicts[j]
                if isinstance(d, dict):
                    if val not in d:
                        d[val] = len(d)
            total_rows += 1

    for i in range(len(col_is_string)):
        if not col_is_string[i]:
            dicts[i] = {}
    return col_names, col_is_string, dicts, total_rows


def _write_dict_blocks(out, dicts, col_is_string):
    for j, is_string in enumerate(col_is_string):
        if not is_string:
            continue
        d = dicts[j] or {}
        M = len(d)
        out.write(struct.pack("<I", M))
        if M == 0:
            continue
        code_to_str = [None] * M
        for s, code in d.items():
            code_to_str[code] = s
        for s in code_to_str:
            if s is None:
                # safeguard; skip missing mapping entries
                b = b""
            else:
                b = s.encode("utf-8")
            out.write(struct.pack("<I", len(b)))
            out.write(b)


def _write_metadata(out, col_names, dicts, col_is_string):
    N = len(col_names)
    out.write(struct.pack("<I", N))
    for j in range(N):
        if col_is_string[j]:
            out.write(struct.pack("<B", 1))
            M = len(dicts[j]) if dicts[j] is not None else 0
            out.write(struct.pack("<I", M))
        else:
            out.write(struct.pack("<B", 0))


def pass2_stream_encode(
    input_path,
    output_path: str,
    col_names,
    col_is_string,
    dicts,
    batch_size: int = 100000,
    delimiter: str = ",",
    quotechar: str = '"',
):
    string_cols = [i for i, flag in enumerate(col_is_string) if flag]
    numeric_cols = [i for i, flag in enumerate(col_is_string) if not flag]

    ncol = len(col_names)
    dicts_for_cols = {
        i: (dicts[i] if dicts[i] is not None else {}) for i in string_cols
    }

    total_rows_written = 0
    start_time = time.time()

    with (
        open(input_path, "r", newline="", encoding="utf-8") as fin,
        open(output_path, "wb") as fout,
    ):
        reader = csv.reader(fin, delimiter=delimiter, quotechar=quotechar)
        header = next(reader, None)
        if header is None:
            raise ValueError("Input file is empty or missing header")
        fout.write(HEADER_MAGIC)
        fout.write(struct.pack("<I", VERSION))
        fout.write(struct.pack("<I", ncol))

        _write_dict_blocks(fout, dicts, col_is_string)
        _write_metadata(fout, col_names, dicts, col_is_string)

        batch = []
        batch_len = 0
        max_workers = max(1, mp.cpu_count())
        pool = None
        if len(string_cols) > 0:
            pool = mp.Pool(processes=min(max_workers, len(string_cols)))

        # Helper for batch encoding
        while True:
            try:
                row = next(reader)
            except StopIteration:
                break
            if len(row) < ncol:
                row = row + [""] * (ncol - len(row))
            elif len(row) > ncol:
                row = row[:ncol]
            batch.append(row)
            batch_len += 1
            if batch_len >= batch_size:
                _process_and_write_batch(
                    batch, fout, string_cols, numeric_cols, dicts_for_cols, pool
                )
                total_rows_written += batch_len
                batch = []
                batch_len = 0
                elapsed = time.time() - start_time
                print(
                    f"[data_loader] Progress: {total_rows_written} rows processed, elapsed={elapsed:.2f}s",
                    flush=True,
                )

        # Flush remainder
        if batch_len > 0:
            _process_and_write_batch(
                batch, fout, string_cols, numeric_cols, dicts_for_cols, pool
            )
            total_rows_written += batch_len
            elapsed = time.time() - start_time
            print(
                f"[data_loader] Final batch: {batch_len} rows, total={total_rows_written}, elapsed={elapsed:.2f}s",
                flush=True,
            )

        if pool is not None:
            pool.close()
            pool.join()

    return total_rows_written


def _process_and_write_batch(
    batch, fout, string_cols, numeric_cols, dicts_for_cols, pool
):
    """Process a single batch: compute string codes (in parallel if pool provided) and write binary row-wise."""
    # Encoding for string columns (parallel if pool available)
    codes_per_col = {}
    if string_cols:
        tasks = []
        for col in string_cols:
            vals = [row[col] for row in batch]
            mapping = dicts_for_cols[col]
            tasks.append((mapping, vals))
        if pool is not None:
            try:
                results = pool.map(_encode_column, tasks)
            except Exception:
                results = []
                for mapping, vals in tasks:
                    codes = [mapping.get(v, -1) for v in vals]
                    results.append(codes)
        else:
            results = []
            for mapping, vals in tasks:
                codes = [mapping.get(v, -1) for v in vals]
                results.append(codes)
        for idx, col in enumerate(string_cols):
            codes_per_col[col] = results[idx]

    # Numeric values by column (float)
    numeric_values_by_col = []
    for col in numeric_cols:
        vals = []
        for row in batch:
            v = row[col]
            if v == "" or v is None:
                vals.append(float("nan"))
            else:
                try:
                    vals.append(float(v))
                except Exception:
                    vals.append(float("nan"))
        numeric_values_by_col.append(vals)

    # Write row-major: string codes then numeric values
    ba = bytearray()
    batch_len = len(batch)
    for i in range(batch_len):
        # string columns in order
        for col in string_cols:
            code = codes_per_col[col][i] if col in codes_per_col else 0
            ba.extend(struct.pack("<i", int(code)))
        # numeric columns in order
        for k in range(len(numeric_cols)):
            val = numeric_values_by_col[k][i]
            ba.extend(struct.pack("<d", float(val)))
    fout.write(ba)


def main():
    parser = argparse.ArgumentParser(description="FaScalSQL data loader")
    parser.add_argument("-i", "--input", required=False, help="Input CSV data file")
    parser.add_argument(
        "-o", "--output", required=False, help="Output binary file or output directory"
    )
    parser.add_argument("--delimiter", default=",", help="CSV delimiter")
    parser.add_argument("--quotechar", default='"', help="CSV quote character")
    parser.add_argument(
        "--batch-size", type=int, default=100000, help="Batch size for streaming"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of rows to process (Pass 1)",
    )
    parser.add_argument(
        "--schema",
        choices=["ssb", "tpch"],
        help="Schema mode for table-to-bin pipeline",
    )
    parser.add_argument(
        "--sf", type=float, default=1.0, help="Scale factor used in schema mode"
    )
    parser.add_argument(
        "--input-dir", help="Input directory for schema mode (e.g., SSB .tbl dir)"
    )
    parser.add_argument(
        "--meta-output",
        default=None,
        help="Optional decode meta output JSON path in schema mode",
    )
    args = parser.parse_args()

    if args.schema:
        if not args.output:
            parser.error("--output is required when --schema is used")
        script_dir = os.path.dirname(os.path.abspath(__file__))
        if args.schema == "ssb":
            src_dir = args.input_dir or os.environ.get(
                "FASCALSQL_SSB_TBL_DIR", _DEFAULT_SSB_TBL_DIR
            )
            ssb_meta_out = args.meta_output or os.environ.get(
                "FASCALSQL_SSB_META_OUT", "fascalsql/configs/verification/ssb_decode_meta.json"
            )
            cmd = [
                "python",
                os.path.join(script_dir, "prepare_fascalsql_ssb_bin.py"),
                "--input-dir",
                src_dir,
                "--output-dir",
                args.output,
                "--meta-output",
                ssb_meta_out,
            ]
        else:
            sf_tag = str(args.sf).replace(".", "")
            tpch_tbl_dir = args.input_dir or os.environ.get(
                "FASCALSQL_TPCH_TBL_DIR", os.path.join(_REPO_ROOT, "tmp", f"fascalsql_tpch_sf{sf_tag}")
            )
            tpch_meta_out = args.meta_output or os.environ.get(
                "FASCALSQL_TPCH_META_OUT", "fascalsql/configs/verification/tpch_decode_meta.json"
            )
            lineitem_tbl = os.path.join(tpch_tbl_dir, "lineitem.tbl")
            if not os.path.exists(lineitem_tbl):
                gen_cmd = [
                    "python",
                    os.path.join(script_dir, "prepare_dogqc_tpch_data.py"),
                    "--sf",
                    str(args.sf),
                    "--output-dir",
                    tpch_tbl_dir,
                ]
                rc = subprocess.call(gen_cmd)
                if rc != 0:
                    return rc
            cmd = [
                "python",
                os.path.join(script_dir, "prepare_fascalsql_tpch_bin.py"),
                "--input-dir",
                tpch_tbl_dir,
                "--output-dir",
                args.output,
                "--meta-output",
                tpch_meta_out,
            ]
        return subprocess.call(cmd)

    if not args.input or not args.output:
        parser.error("--input and --output are required unless --schema is used")

    input_path = args.input
    output_path = args.output
    delimiter = args.delimiter
    quotechar = args.quotechar
    batch_size = args.batch_size

    print("Pass 1: analyzing input to build dictionaries...")
    col_names, col_is_string, dicts, total_rows = pass1_build_dicts(
        input_path, delimiter=delimiter, quotechar=quotechar
    )
    print(
        f"Pass 1 complete. Rows={total_rows}, Columns={len(col_names)}, String cols={sum(col_is_string)}"
    )

    print("Pass 2: streaming encode and write binary...")
    written = pass2_stream_encode(
        input_path,
        output_path,
        col_names,
        col_is_string,
        dicts,
        batch_size=batch_size,
        delimiter=delimiter,
        quotechar=quotechar,
    )
    print(f"Pass 2 complete. Rows written: {written}")


if __name__ == "__main__":
    main()


# ============================================================
# FaScalSQL Zero-Copy Pinned Column Manager
# Calls column_alloc.cu functions via ctypes for pinned GPU-accessible memory
# ============================================================

import ctypes
import numpy as np
from typing import Dict, Optional, List


class PinnedColumnManager:
    """
    Manages pinned (page-locked) column memory for zero-copy GPU access.
    Wraps fascal_pin_columns() from column_alloc.cu.
    """

    def __init__(self, lib_path: Optional[str] = None):
        self._lib = None
        self._pinned = {}  # col_name -> (host_ptr, dev_ptr, size_bytes)
        if lib_path:
            try:
                self._lib = ctypes.CDLL(lib_path)
                self._setup_lib()
            except Exception as e:
                print(f"[PinnedColumnManager] Could not load {lib_path}: {e}")
                print("[PinnedColumnManager] Will use numpy arrays (no pinning)")

    def _setup_lib(self):
        if self._lib is None:
            return
        self._lib.fascal_pin_columns.restype = ctypes.c_int
        self._lib.fascal_pin_columns.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),  # host_ptrs
            ctypes.POINTER(ctypes.c_void_p),  # dev_ptrs
            ctypes.POINTER(ctypes.c_size_t),  # sizes_bytes
            ctypes.c_int,  # n_cols
        ]
        self._lib.fascal_unpin_columns.restype = None
        self._lib.fascal_unpin_columns.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_int,
        ]

    def pin_numpy_columns(self, columns: Dict[str, np.ndarray]) -> Dict[str, dict]:
        """
        Pin numpy arrays as mapped host memory.
        Returns metadata dict: col_name -> {host_ptr, dev_ptr, size_bytes, is_pinned}
        """
        result = {}
        if self._lib is None:
            # No CUDA lib available — return metadata with numpy pointers only
            for name, arr in columns.items():
                arr_contiguous = np.ascontiguousarray(arr, dtype=np.int32)
                result[name] = {
                    "host_ptr": arr_contiguous.ctypes.data,
                    "dev_ptr": None,
                    "size_bytes": arr_contiguous.nbytes,
                    "is_pinned": False,
                    "array": arr_contiguous,
                }
            return result

        names = list(columns.keys())
        arrs = [np.ascontiguousarray(columns[n], dtype=np.int32) for n in names]
        n = len(names)

        host_ptrs = (ctypes.c_void_p * n)(*[a.ctypes.data for a in arrs])
        dev_ptrs = (ctypes.c_void_p * n)()
        sizes = (ctypes.c_size_t * n)(*[a.nbytes for a in arrs])

        failures = self._lib.fascal_pin_columns(host_ptrs, dev_ptrs, sizes, n)
        if failures > 0:
            print(
                f"[PinnedColumnManager] Warning: {failures}/{n} columns failed to pin"
            )

        for i, name in enumerate(names):
            result[name] = {
                "host_ptr": int(host_ptrs[i]),
                "dev_ptr": int(dev_ptrs[i]) if dev_ptrs[i] else None,
                "size_bytes": int(sizes[i]),
                "is_pinned": (dev_ptrs[i] is not None),
                "array": arrs[i],
            }
            self._pinned[name] = result[name]

        return result

    def unpin_all(self):
        """Unpin all registered columns."""
        if self._lib is None or not self._pinned:
            return
        names = list(self._pinned.keys())
        ptrs = (ctypes.c_void_p * len(names))(
            *[self._pinned[n]["host_ptr"] for n in names]
        )
        self._lib.fascal_unpin_columns(ptrs, len(names))
        self._pinned.clear()

    def __del__(self):
        try:
            self.unpin_all()
        except Exception:
            pass


def load_columns_pinned(
    file_paths: Dict[str, str],
    num_tuples: int,
    lib_path: Optional[str] = None,
) -> Dict[str, dict]:
    """
    Load binary int32 columns and pin them for zero-copy GPU access.

    Args:
        file_paths: dict of col_name -> .bin file path
        num_tuples: number of tuples in each column
        lib_path: path to libcolumn_alloc.so (optional; falls back to plain numpy)

    Returns:
        dict of col_name -> {host_ptr, dev_ptr, size_bytes, is_pinned, array}
    """
    columns = {}
    for name, path in file_paths.items():
        try:
            arr = np.fromfile(path, dtype=np.int32)
            if len(arr) < num_tuples:
                print(
                    f"[load_columns_pinned] Warning: {path} has {len(arr)} rows, expected {num_tuples}"
                )
            columns[name] = arr[:num_tuples]
        except Exception as e:
            print(f"[load_columns_pinned] Error loading {path}: {e}")
            columns[name] = np.zeros(num_tuples, dtype=np.int32)

    mgr = PinnedColumnManager(lib_path)
    return mgr.pin_numpy_columns(columns)


def load_columns_chunked_pinned(
    file_paths: Dict[str, str],
    total_tuples: int,
    chunk_size: int = 10_000_000,
    lib_path: Optional[str] = None,
) -> List[Dict[str, dict]]:
    """
    Load large binary columns in chunks (for SF=100 where full pinning may fail).
    Returns a list of chunk metadata dicts.

    Each chunk dict: col_name -> {host_ptr, dev_ptr, size_bytes, is_pinned, array}
    """
    chunks = []
    offset = 0
    while offset < total_tuples:
        count = min(chunk_size, total_tuples - offset)
        chunk_cols = {}
        for name, path in file_paths.items():
            try:
                arr = np.fromfile(path, dtype=np.int32, count=count, offset=offset * 4)
                chunk_cols[name] = arr
            except Exception as e:
                print(f"[load_columns_chunked_pinned] Error chunk offset={offset}: {e}")
                chunk_cols[name] = np.zeros(count, dtype=np.int32)
        mgr = PinnedColumnManager(lib_path)
        chunk_meta = mgr.pin_numpy_columns(chunk_cols)
        chunks.append(chunk_meta)
        offset += count
    return chunks
