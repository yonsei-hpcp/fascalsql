#!/usr/bin/env python3
"""Fast parallel lineitem.tbl → .bin converter.

Strategy: split file into N chunks by byte offset, each worker processes
independently (no IPC), then concatenate the per-worker .bin files.

Dict columns (returnflag=3, linestatus=2, shipinstruct=4, shipmode=7)
are tiny and pre-populated so no cross-worker coordination needed.

Usage:
  python3 fast_tpch_lineitem_convert.py /tmp/tpch_tbl_sf100/lineitem.tbl /tmp/fascalsql_tpch_data_sf100/
"""
import sys, json, time, os, struct
from pathlib import Path
from multiprocessing import Process, cpu_count
import numpy as np

TBL_COLS = [
    "l_orderkey", "l_partkey", "l_suppkey", "l_linenumber",
    "l_quantity", "l_extendedprice", "l_discount", "l_tax",
    "l_returnflag", "l_linestatus", "l_shipdate", "l_commitdate",
    "l_receiptdate", "l_shipinstruct", "l_shipmode", "l_comment",
]

# Columns to emit
# l_quantity (idx 4) is DECIMAL(15,2) — stored as float32 to match prepare_fascalsql_tpch_bin.py
INT_IDXS = [0, 1, 2, 3]
FLOAT_IDXS = [4, 5, 6, 7]
DATE_IDXS = [10, 11, 12]
DICT_IDXS = [8, 9, 13, 14]  # skip l_comment (idx 15)

# Pre-populated dicts for lineitem (tiny, fixed across all SF)
PRESET_DICTS = {
    8: {"A": 0, "N": 1, "R": 2},  # l_returnflag
    9: {"F": 0, "O": 1},  # l_linestatus
    13: {"DELIVER IN PERSON": 0, "COLLECT COD": 1, "NONE": 2, "TAKE BACK RETURN": 3},
    14: {"REG AIR": 0, "AIR": 1, "RAIL": 2, "SHIP": 3, "TRUCK": 4, "MAIL": 5, "FOB": 6},
}

EMIT_COLS = INT_IDXS + FLOAT_IDXS + DATE_IDXS + DICT_IDXS  # all non-skipped
BATCH = 500_000


def find_chunk_offsets(filepath, nchunks):
    """Find byte offsets that split the file into ~equal chunks at line boundaries."""
    fsize = os.path.getsize(filepath)
    offsets = [0]
    with open(filepath, "rb") as f:
        for i in range(1, nchunks):
            target = fsize * i // nchunks
            f.seek(target)
            f.readline()  # skip to next line boundary
            offsets.append(f.tell())
    offsets.append(fsize)
    return offsets


def worker(filepath, start_off, end_off, worker_id, tmp_dir):
    """Process a chunk of the file from start_off to end_off bytes."""
    int_bufs = {i: [] for i in INT_IDXS + DATE_IDXS + DICT_IDXS}
    float_bufs = {i: [] for i in FLOAT_IDXS}

    # Open per-worker temp files
    fps_int = {}
    fps_float = {}
    for i in INT_IDXS + DATE_IDXS + DICT_IDXS:
        fps_int[i] = open(tmp_dir / f"w{worker_id}_{TBL_COLS[i]}.bin", "wb")
    for i in FLOAT_IDXS:
        fps_float[i] = open(tmp_dir / f"w{worker_id}_{TBL_COLS[i]}.bin", "wb")

    dicts = {i: dict(d) for i, d in PRESET_DICTS.items()}
    rows = 0

    with open(filepath, "r") as f:
        f.seek(start_off)
        while f.tell() < end_off:
            line = f.readline()
            if not line:
                break
            parts = line.split("|")

            for i in INT_IDXS:
                int_bufs[i].append(int(parts[i]))
            for i in FLOAT_IDXS:
                float_bufs[i].append(float(parts[i]))
            for i in DATE_IDXS:
                s = parts[i]
                int_bufs[i].append(int(s[0:4]) * 10000 + int(s[5:7]) * 100 + int(s[8:10]))
            for i in DICT_IDXS:
                raw = parts[i]
                d = dicts[i]
                if raw not in d:
                    d[raw] = len(d)
                int_bufs[i].append(d[raw])

            rows += 1
            if rows % BATCH == 0:
                for i in INT_IDXS + DATE_IDXS + DICT_IDXS:
                    np.array(int_bufs[i], dtype=np.int32).tofile(fps_int[i])
                    int_bufs[i].clear()
                for i in FLOAT_IDXS:
                    np.array(float_bufs[i], dtype=np.float32).tofile(fps_float[i])
                    float_bufs[i].clear()

    # Flush
    for i in INT_IDXS + DATE_IDXS + DICT_IDXS:
        if int_bufs[i]:
            np.array(int_bufs[i], dtype=np.int32).tofile(fps_int[i])
    for i in FLOAT_IDXS:
        if float_bufs[i]:
            np.array(float_bufs[i], dtype=np.float32).tofile(fps_float[i])

    for fp in fps_int.values():
        fp.close()
    for fp in fps_float.values():
        fp.close()

    # Write row count
    (tmp_dir / f"w{worker_id}_count.txt").write_text(str(rows))

    # Write any new dict entries discovered
    for i in DICT_IDXS:
        with open(tmp_dir / f"w{worker_id}_dict_{i}.json", "w") as f:
            json.dump(dicts[i], f)


def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <lineitem.tbl> <output_dir>")
        sys.exit(1)

    tbl_path = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])
    output_dir.mkdir(parents=True, exist_ok=True)

    nworkers = min(16, max(1, cpu_count() - 2))
    tmp_dir = output_dir / "_tmp_lineitem"
    tmp_dir.mkdir(exist_ok=True)

    t0 = time.time()
    print(f"Splitting {tbl_path} into {nworkers} chunks ...")
    offsets = find_chunk_offsets(str(tbl_path), nworkers)
    print(f"  Chunk offsets computed in {time.time()-t0:.1f}s")

    # Launch workers
    print(f"Launching {nworkers} workers ...")
    procs = []
    for w in range(nworkers):
        p = Process(target=worker, args=(str(tbl_path), offsets[w], offsets[w + 1], w, tmp_dir))
        p.start()
        procs.append(p)

    # Wait for all
    for p in procs:
        p.join()

    t1 = time.time()
    total_rows = sum(int((tmp_dir / f"w{w}_count.txt").read_text()) for w in range(nworkers))
    print(f"  All workers done: {total_rows:,} rows in {t1-t0:.1f}s ({total_rows/(t1-t0)/1e6:.2f}M/s)")

    # Concatenate per-worker files
    print("Concatenating ...")
    all_cols = INT_IDXS + FLOAT_IDXS + DATE_IDXS + DICT_IDXS
    for i in all_cols:
        col = TBL_COLS[i]
        with open(output_dir / f"{col}.bin", "wb") as out:
            for w in range(nworkers):
                chunk_file = tmp_dir / f"w{w}_{col}.bin"
                with open(chunk_file, "rb") as inp:
                    while True:
                        buf = inp.read(64 * 1024 * 1024)  # 64MB chunks
                        if not buf:
                            break
                        out.write(buf)
                chunk_file.unlink()

    # _enc.bin copies
    import shutil
    print("Writing _enc.bin ...")
    for i in DICT_IDXS + DATE_IDXS:
        col = TBL_COLS[i]
        shutil.copy2(output_dir / f"{col}.bin", output_dir / f"{col}_enc.bin")

    # Merge dicts
    print("Merging dicts ...")
    for i in DICT_IDXS:
        merged = {}
        for w in range(nworkers):
            dp = tmp_dir / f"w{w}_dict_{i}.json"
            with open(dp) as f:
                d = json.load(f)
            for k, v in d.items():
                if k not in merged:
                    merged[k] = len(merged)
            dp.unlink()
        col = TBL_COLS[i]
        rev = {str(v): k for k, v in merged.items()}
        with open(output_dir / f"{col}_dict.json", "w") as f:
            json.dump(rev, f, indent=2)
        print(f"  {col}: {len(merged)} entries")

    # Cleanup
    with open(output_dir / "l_comment_dict.json", "w") as f:
        json.dump({}, f)
    (output_dir / "lineitem_count.txt").write_text(str(total_rows))
    for f in tmp_dir.iterdir():
        f.unlink()
    tmp_dir.rmdir()

    elapsed = time.time() - t0
    print(f"\nDone! {total_rows:,} rows in {elapsed:.1f}s ({total_rows/elapsed/1e6:.2f}M/s)")


if __name__ == "__main__":
    main()
