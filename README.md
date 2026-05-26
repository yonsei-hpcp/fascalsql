# FaScalSQL [ICDE '26]

This repository contains the source code for [FaScalSQL \[ICDE '26\]](https://yonsei-hpcp.github.io/files/fascalsql-icde2026-preprint.pdf), a fast and scalable GPU-accelerated analytical SQL query engine for out-of-memory tables.
If you find FaScalSQL useful to your research, please cite:

```bibtex
@inproceedings{lim2026fascalsql,
  author    = {Chaemin Lim and Suhyun Lee and Jinwoo Choi and Kwanghyun Park and Jinho Lee and Joonsung Kim and Youngsok Kim},
  title     = {{FaScalSQL: A Fast and Scalable GPU-Accelerated SQL Query Engine for Out-of-Memory Tables}},
  booktitle = {Proc. 42nd IEEE International Conference on Data Engineering (ICDE)},
  year      = {2026},
}
```

## Prerequisites

| Requirement | Version |
|-------------|---------|
| Linux + NVIDIA GPU | Compute capability 7.0+ (tested: RTX A4000) |
| CUDA Toolkit | 12.x (`nvcc`, runtime) |
| GCC | 9+ with C++14, AVX2 support |
| Python | 3.10+ |
| Python packages | `pip install -r requirements.txt` |
| GNU Make | any |

## Quick Start

```bash
git clone --recursive https://github.com/yonsei-hpcp/fascalsql.git
cd fascalsql
pip install -r requirements.txt
cd fascalsql/host && make standalone_all -j$(nproc) && cd ../..
```

### Generate SF=1 Data

```bash
# Generate TPC-H .tbl
cd third_party/tpch-dbgen && make -j4 && ./dbgen -vf -s 1 && cd ../..

# Generate SSB .tbl
cd third_party/ssb-dbgen && make -j4 && ./dbgen -vf -s 1 -T a && cd ../..

# Convert to binary columns
export PYTHONPATH=$PWD:$PYTHONPATH

python3 fascalsql/python/data/prepare_fascalsql_tpch_bin.py \
  --input-dir third_party/tpch-dbgen --output-dir tmp/fascalsql_tpch_data

python3 fascalsql/python/data/fast_tpch_lineitem_convert.py \
  third_party/tpch-dbgen/lineitem.tbl tmp/fascalsql_tpch_data/

python3 fascalsql/python/data/prepare_fascalsql_ssb_bin.py \
  --input-dir third_party/ssb-dbgen --output-dir tmp/fascalsql_ssb_data

python3 fascalsql/python/data/fast_ssb_lineorder_convert.py \
  third_party/ssb-dbgen/lineorder.tbl tmp/fascalsql_ssb_data/

# Create DuckDB databases from .tbl files
python3 fascalsql/python/utils/create_duckdb_from_tbl.py \
  --benchmark tpch --tbl-dir third_party/tpch-dbgen --output tmp/tpch.duckdb

python3 fascalsql/python/utils/create_duckdb_from_tbl.py \
  --benchmark ssb --tbl-dir third_party/ssb-dbgen --output tmp/ssb.duckdb

# Generate golden references
python3 fascalsql/python/utils/generate_golden_from_duckdb.py \
  --benchmark tpch --duckdb-db tmp/tpch.duckdb
python3 fascalsql/python/utils/generate_golden_from_duckdb.py \
  --benchmark ssb --duckdb-db tmp/ssb.duckdb
```

### Run a Query

```bash
cd fascalsql/host

# TPC-H Q6
FASCALSQL_DATA_DIR=tmp/fascalsql_tpch_data \
  build/standalone/tpch_q6 0 tmp/fascalsql_tpch_data

# SSB Q3.1
FASCALSQL_DATA_DIR=tmp/fascalsql_ssb_data \
  build/standalone/ssb_q31 0 tmp/fascalsql_ssb_data
```

### Verify All Queries

```bash
python3 fascalsql/python/utils/verify_tpch_results.py \
  --data-dir tmp/fascalsql_tpch_data --duckdb-db tmp/tpch.duckdb \
  --result-schema fascalsql/configs/verification/tpch_result_schema.json

python3 fascalsql/python/utils/verify_ssb_results.py \
  --data-dir tmp/fascalsql_ssb_data --duckdb-db tmp/ssb.duckdb \
  --schema fascalsql/configs/verification/ssb_result_schema.json
```

## Reproduce Paper Results (SF=100)

### 1. Generate SF=100 Data

```bash
cd third_party/tpch-dbgen && ./dbgen -vf -s 100 && cd ../..
cd third_party/ssb-dbgen && ./dbgen -vf -s 100 -T a && cd ../..

export PYTHONPATH=$PWD:$PYTHONPATH

python3 fascalsql/python/data/prepare_fascalsql_tpch_bin.py \
  --input-dir third_party/tpch-dbgen --output-dir tmp/fascalsql_tpch_data_sf100

python3 fascalsql/python/data/fast_tpch_lineitem_convert.py \
  third_party/tpch-dbgen/lineitem.tbl tmp/fascalsql_tpch_data_sf100/

python3 fascalsql/python/data/prepare_fascalsql_ssb_bin.py \
  --input-dir third_party/ssb-dbgen --output-dir tmp/fascalsql_ssb_data_sf100

python3 fascalsql/python/data/fast_ssb_lineorder_convert.py \
  third_party/ssb-dbgen/lineorder.tbl tmp/fascalsql_ssb_data_sf100/
```

### 2. Run Benchmarks

Run individual queries and collect timing. Each binary prints `t_query` (ms):

```bash
cd fascalsql/host

# Example: TPC-H Q6
build/standalone/tpch_q6 0 tmp/fascalsql_tpch_data_sf100

# Example: SSB Q3.1
build/standalone/ssb_q31 0 tmp/fascalsql_ssb_data_sf100
```

Repeat with ablation flags to reproduce Figure 12 (ODZC vs +AFP vs +CQO).

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `FASCALSQL_DATA_DIR` | Data directory for .bin files | (pass as argv) |
| `FASCAL_BLOOM_OFF` | Disable CPU Bloom filters | unset (enabled) |
| `FASCAL_GPU_BATCH` | GPU batch size (tuples/stream) | 4194304 |
| `FASCAL_CPU_THREADS` | CPU thread count for AFP | hardware_concurrency |
| `FASCAL_REPROFILE` | Force hardware re-profiling | unset |

## Repository Layout

```
fascalsql/
  kernels/           # CUDA kernels (35 queries)
  configs/
    ablation/        # Per-query ablation configs
    pipelines/       # Per-query pipeline configs
    verification/    # Golden references, schemas
  host/
    include/         # C++/CUDA headers (ODZC, AFP, CQO)
    src/             # Runtime implementation
    Makefile
  python/
    data/            # .tbl -> .bin converters
    utils/           # Verification & encoding

```

No `scripts/` directory is included; queries are run directly via the standalone binaries.

```
third_party/
  tpch-dbgen/        # TPC-H data generator
  ssb-dbgen/         # SSB data generator
```

## License

Apache License 2.0. See [LICENSE](LICENSE).
