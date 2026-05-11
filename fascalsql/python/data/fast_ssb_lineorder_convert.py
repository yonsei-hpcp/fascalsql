#!/usr/bin/env python3
"""Fast parallel lineorder.tbl → .bin converter for SSB.
File-split strategy: split by byte offset, each worker independent.

Usage:
  python3 fast_ssb_lineorder_convert.py /tmp/ssb_tbl_sf100/lineorder.tbl /tmp/fascalsql_ssb_data_sf100/
"""
import sys, json, time, os
from pathlib import Path
from multiprocessing import Process, cpu_count
import numpy as np

# SSB lineorder .tbl field order (17 fields per line)
# Index: Name               Type
#   0    lo_orderkey         int
#   1    lo_linenumber       int
#   2    lo_custkey          int
#   3    lo_partkey          int
#   4    lo_suppkey          int
#   5    lo_orderdate        int (YYYYMMDD)
#   6    lo_orderpriority    string (SKIP)
#   7    lo_shippriority     int
#   8    lo_quantity         int
#   9    lo_extendedprice    int
#  10    lo_ordtotalprice    int
#  11    lo_discount         int
#  12    lo_revenue          int
#  13    lo_supplycost       int
#  14    lo_tax              int
#  15    lo_commitdate       int (YYYYMMDD)
#  16    lo_shipmode         string (SKIP)

INT_FIELDS = [
    (0, "lo_orderkey"), (1, "lo_linenumber"), (2, "lo_custkey"),
    (3, "lo_partkey"), (4, "lo_suppkey"), (5, "lo_orderdate"),
    (7, "lo_shippriority"), (8, "lo_quantity"), (9, "lo_extendedprice"),
    (10, "lo_ordtotalprice"), (11, "lo_discount"), (12, "lo_revenue"),
    (13, "lo_supplycost"), (14, "lo_tax"), (15, "lo_commitdate"),
]

BATCH = 500_000


def find_chunk_offsets(filepath, nchunks):
    fsize = os.path.getsize(filepath)
    offsets = [0]
    with open(filepath, "rb") as f:
        for i in range(1, nchunks):
            f.seek(fsize * i // nchunks)
            f.readline()
            offsets.append(f.tell())
    offsets.append(fsize)
    return offsets


def worker(filepath, start_off, end_off, worker_id, tmp_dir):
    nfields = len(INT_FIELDS)
    bufs = [[] for _ in range(nfields)]
    fps = [open(tmp_dir / f"w{worker_id}_{col}.bin", "wb") for _, col in INT_FIELDS]

    rows = 0
    with open(filepath, "r") as f:
        f.seek(start_off)
        while f.tell() < end_off:
            line = f.readline()
            if not line:
                break
            parts = line.split("|")
            for j, (idx, _) in enumerate(INT_FIELDS):
                bufs[j].append(int(parts[idx]))
            rows += 1
            if rows % BATCH == 0:
                for j in range(nfields):
                    np.array(bufs[j], dtype=np.int32).tofile(fps[j])
                    bufs[j].clear()

    for j in range(nfields):
        if bufs[j]:
            np.array(bufs[j], dtype=np.int32).tofile(fps[j])
    for fp in fps:
        fp.close()
    (tmp_dir / f"w{worker_id}_count.txt").write_text(str(rows))


def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <lineorder.tbl> <output_dir>")
        sys.exit(1)

    tbl_path = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])
    output_dir.mkdir(parents=True, exist_ok=True)

    nworkers = min(16, max(1, cpu_count() - 2))
    tmp_dir = output_dir / "_tmp_lineorder"
    if tmp_dir.exists():
        import shutil
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir()

    t0 = time.time()
    print(f"Splitting {tbl_path} into {nworkers} chunks ...")
    offsets = find_chunk_offsets(str(tbl_path), nworkers)

    print(f"Launching {nworkers} workers ...")
    procs = []
    for w in range(nworkers):
        p = Process(target=worker, args=(str(tbl_path), offsets[w], offsets[w+1], w, tmp_dir))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()

    total_rows = sum(int((tmp_dir / f"w{w}_count.txt").read_text()) for w in range(nworkers))
    t1 = time.time()
    print(f"  Workers done: {total_rows:,} rows in {t1-t0:.1f}s ({total_rows/(t1-t0)/1e6:.2f}M/s)")

    print("Concatenating ...")
    for _, col in INT_FIELDS:
        with open(output_dir / f"{col}.bin", "wb") as out:
            for w in range(nworkers):
                cf = tmp_dir / f"w{w}_{col}.bin"
                with open(cf, "rb") as inp:
                    while True:
                        buf = inp.read(64 * 1024 * 1024)
                        if not buf:
                            break
                        out.write(buf)
                cf.unlink()

    import shutil
    for col in ["lo_orderdate", "lo_commitdate"]:
        shutil.copy2(output_dir / f"{col}.bin", output_dir / f"{col}_enc.bin")

    (output_dir / "lineorder_count.txt").write_text(str(total_rows))
    for f in tmp_dir.iterdir():
        f.unlink()
    tmp_dir.rmdir()

    elapsed = time.time() - t0
    print(f"\nDone! {total_rows:,} rows in {elapsed:.1f}s ({total_rows/elapsed/1e6:.2f}M/s)")


if __name__ == "__main__":
    main()
