#!/usr/bin/env python3
"""Fast streaming converter for TPC-H customer.tbl.

The generic DDL-driven converter builds Python dictionaries for every string
column.  At SF=100, customer strings are effectively row-unique and that
approach is unnecessarily memory-heavy.  This converter uses the customer row
index as the encoded string id and emits the runtime dictionary files needed by
the standalone kernels.
"""

from __future__ import annotations

import json
import struct
import sys
import time
from pathlib import Path

import numpy as np


MKTSEGMENT = {
    "AUTOMOBILE": 0,
    "BUILDING": 1,
    "FURNITURE": 2,
    "HOUSEHOLD": 3,
    "MACHINERY": 4,
}
BATCH = 500_000


def parse_decimal_2(raw: str) -> int:
    return int(round(float(raw) * 100.0))


def flush_int(bufs: dict[str, list[int]], fps: dict[str, object]) -> None:
    for col, vals in bufs.items():
        if vals:
            np.asarray(vals, dtype=np.int32).tofile(fps[col])
            vals.clear()


def main() -> int:
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <customer.tbl> <output_dir>")
        return 1

    tbl_path = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])
    output_dir.mkdir(parents=True, exist_ok=True)

    int_cols = [
        "c_custkey",
        "c_name",
        "c_address",
        "c_nationkey",
        "c_phone",
        "c_acctbal",
        "c_mktsegment",
        "c_comment",
        "c_custkey_enc",
        "c_nationkey_enc",
        "c_mktsegment_enc",
    ]
    fps = {col: (output_dir / f"{col}.bin").open("wb") for col in int_cols}
    dict_cols = ("c_address", "c_phone", "c_comment", "c_mktsegment")
    dict_fps = {
        col: (output_dir / f"customer_{col}.dict").open("w", encoding="utf-8")
        for col in dict_cols
    }
    phone_offsets = (output_dir / "customer_c_phone_dict_offsets.bin").open("wb")
    phone_strings = (output_dir / "customer_c_phone_dict_strings.bin").open("wb")

    bufs = {col: [] for col in int_cols}
    rows = 0
    phone_offset = 0
    t0 = time.time()
    try:
        with tbl_path.open("r", encoding="utf-8") as fin:
            for line in fin:
                parts = line.rstrip("\n").split("|")
                if parts and parts[-1] == "":
                    parts.pop()
                row_id = rows
                custkey = int(parts[0])
                mkt = MKTSEGMENT[parts[6]]

                bufs["c_custkey"].append(custkey)
                bufs["c_name"].append(row_id)
                bufs["c_address"].append(row_id)
                bufs["c_nationkey"].append(int(parts[3]))
                bufs["c_phone"].append(row_id)
                bufs["c_acctbal"].append(parse_decimal_2(parts[5]))
                bufs["c_mktsegment"].append(mkt)
                bufs["c_comment"].append(row_id)
                bufs["c_custkey_enc"].append(custkey - 1)
                bufs["c_nationkey_enc"].append(int(parts[3]))
                bufs["c_mktsegment_enc"].append(mkt)

                dict_fps["c_address"].write(f"{row_id}|{parts[2]}\n")
                dict_fps["c_phone"].write(f"{row_id}|{parts[4]}\n")
                dict_fps["c_comment"].write(f"{row_id}|{parts[7]}\n")
                if row_id < len(MKTSEGMENT):
                    # Written below after the loop in stable encoded-id order.
                    pass

                phone_offsets.write(struct.pack("<i", phone_offset))
                phone_bytes = parts[4].encode("utf-8") + b"\0"
                phone_strings.write(phone_bytes)
                phone_offset += len(phone_bytes)

                rows += 1
                if rows % BATCH == 0:
                    flush_int(bufs, fps)
                    print(f"  customer_rows={rows}")
        flush_int(bufs, fps)
    finally:
        for fp in fps.values():
            fp.close()
        for fp in dict_fps.values():
            fp.close()
        phone_offsets.close()
        phone_strings.close()

    (output_dir / "customer_c_mktsegment.dict").write_text(
        "\n".join(f"{v}|{k}" for k, v in sorted(MKTSEGMENT.items(), key=lambda kv: kv[1])) + "\n",
        encoding="utf-8",
    )
    (output_dir / "c_mktsegment_dict.json").write_text(
        json.dumps(MKTSEGMENT, indent=2),
        encoding="utf-8",
    )
    (output_dir / "customer_count.txt").write_text(f"{rows}\n")
    elapsed = time.time() - t0
    print(f"Done customer: {rows:,} rows in {elapsed:.1f}s ({rows / elapsed / 1e6:.2f}M/s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
