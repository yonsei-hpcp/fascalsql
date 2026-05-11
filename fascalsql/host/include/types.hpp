#pragma once

#include <vector>
#include <string>
#include <iostream>
#include <cstdio>
#include <map>

#define COMPRESSION_16 1    // 16-bit compression, 32 elements in 512-bit
#define COMPRESSION_32 2    // 32-bit compression, 16 elements in 512-bit
#define PIPELINE_DEPTH 4
#define STORAGE_PAGE_SIZE ((size_t)1024 * 4)
#define NUM_ELEM_IN_PAGE (STORAGE_PAGE_SIZE / sizeof(int))
#define BLKWISE_BLOCK_SIZE ((size_t)1024 * 1024 * 128)
#define ELEM_IN_BLKWISE_BLK (BLKWISE_BLOCK_SIZE / sizeof(int))
#define NUM_PAGE_IN_BLK (BLKWISE_BLOCK_SIZE / (STORAGE_PAGE_SIZE))
#define BITMAP_1BIT_SIZE_IN_BYTE ((ELEM_IN_BLKWISE_BLK) / (8))
#define BLOOM_FILTER_SIZE_IN_BYTE ((64) * (4096) * (24) / (8))
#define NUM_MAX_PARTITIONED_BLOOM_FILTER (1)

#define RUNTYPE_SCANPROJ_ONLY 1
#define RUNTYPE_BLOOMFILTER_ONLY 2
#define RUNTYPE_PARTITIONING 3

#define RUNTYPE_SCAN_OFFLOADING_ONLY 4


#define GPU_L1DCACHE_SIZE ((size_t)128 * 1024)

#define NO_PARTITIONING (-1)
#define PARTITIONING_KEY 2

#define TOGB(x) ((x) / (1024 * 1024 * 1024L))
#define GB(x) ((x) * 1024 * 1024 * 1024L)
#define MB(x) ((x) * 1024 * 1024L)
#define MBFLOAT(x) ((x) * 1024 * 1024L * 1.0)
#define KB(x) ((x) * 1024L)

// #include "spdlog/sinks/stdout_color_sinks.h"
#define FASCALSQL_CHECK(call)                                                                        \
    {                                                                                                 \
        Result_t result = call;                                                                       \
        if (result != RESULT_OK)                                                                      \
        {                                                                                             \
            printf("%s:%d Error calling " #call ", error code is: %d\n", __FILE__, __LINE__, result); \
            exit(-1);                                                                                 \
        }                                                                                             \
    }
#define FASCALSQL_NULLCHECK(val)                                                    \
    if (val == nullptr)                                                              \
    {                                                                                \
        printf("%s:%d NULL Pointer Error: calling " #val " \n", __FILE__, __LINE__); \
        exit(-1);                                                                    \
    }

#define FASCALSQL_ASSERT(call)                                                     \
    if (call)                                                                       \
    {                                                                               \
        printf("%s:%d Assertion Failed: calling " #call " \n", __FILE__, __LINE__); \
        exit(-1);                                                                   \
    }

#define FASCALSQL_PREAD_CHECK(call)                                                    \
        if (!call)                                                                       \
        {                                                                               \
            printf("%s:%d Assertion Failed: calling " #call " \n", __FILE__, __LINE__); \
            exit(-1);                                                                   \
        }

#define FASCALSQL_EXIT_SUCCESS 0
#define FASCALSQL_EXIT_FAILURE 1

enum DataType
{
    INT32,
    INT64,
    FLOAT32,
    DOUBLE64,
    CHAR,
    VARCHAR
};

enum Result_t
{
    RESULT_OK = 0,
    RESULT_ERROR_,
    RESULT_ERROR_INVALID_INPUT,
    RESULT_ERROR_NOT_IMPLEMENTED,
    RESULT_ERROR_NOT_ALLOWED,
    RESULT_ERROR_NULLPTR,
};

enum StorageEngineType
{
    AQUOMAN,
};

typedef int64_t table_id_t;
typedef int64_t column_id_t;
typedef int64_t page_id_t;

#define PAGE_SIZE ((ssize_t)1024 * 1024)

typedef struct
{
    int64_t page_id;
    int64_t starting_row_vector_id;
    int32_t num_row_vector;
    int32_t num_records;
    int32_t data_type;
    int32_t data_length;
} AquomanPageHeader_t;

typedef struct
{
    AquomanPageHeader_t aquoman_page_header;
    char buffer[(PAGE_SIZE - sizeof(AquomanPageHeader_t))];
} aquoman_page_t;

/********************************************/

#define SQLOperator_NOP 0
#define SQLOperator_AGGREGATE 1
#define SQLOperator_AGGREGATE_GROUPBY 2
#define SQLOperator_SORT 3
#define SQLOperator_SORT_MERGE 4
#define SQLOperator_TOPK 5

#define DTYPE_INT32 0
#define DTYPE_INT64 1

// SSB TABLES
#define SSB_TABLE_DATE 0
#define SSB_TABLE_SUPPLIER 1 
#define SSB_TABLE_CUSTOMER 2
#define SSB_TABLE_PART 3
#define SSB_TABLE_LINEORDER 4 

#define OK_RUN 1
#define OK_STOP 0

// Hash function types
#define HF_MULT_SHIFT 0
#define HF_MULT_ADD_SHIFT 1
#define HF_MURMUR_2 2
#define HF_MURMUR_3 3
#define HF_TABULATION 4
#define HF_XX 5

// Hashing schemes
#define HS_LINEAR_PROBING 0
#define HS_DOUBLE_PROBING 1
#define HS_QUADRATIC_PROBIN 2
#define HS_ROBIN_HOOD 3
#define HS_CUCKOO 4

// Data distribution types
#define DIST_DENSE 0
#define DIST_UNIFORM 1
#define DIST_GRID 2

typedef struct arguments_t
{
    // Common arguments - for classify
    std::string test_name;
    // ssb query number
    int ssb_qnum;
    // ssb scale factor
    int scale_factor;
    int64_t gpu_streaming_buffer_size;

    int join_type;
    int join_order;
    int motivation_data;
    std::string join_target;
    int pipeline_depth;
    int bitmap_depth;
    int projection_depth;

    int filter_depth;
    std::string hash_table;
    std::string column_1;
    std::string column_2;
    int inhost_type;
    int fascal_type = 0;
    int bloomfiltersize;
    int64_t block_size_byte;
    // Common arguments - device id
    int device_id;   // For single-CSD
    int device_id_0; // For multi-CSD, #0
    int device_id_1; // For multi-CSD, #1
    int device_id_2; // For multi-CSD, #2
    int device_id_3; // For multi-CSD, #3
    int device_id_4; // For multi-CSD, #4
    int device_id_5; // For multi-CSD, #5
    int device_id_6; // For multi-CSD, #6
    int device_id_7; // For multi-CSD, #7

    std::string gpu_execution_mode;
    std::string ssb_header_path;
    int ssb_stroage_num;
    int nvme_dev_num;
    std::vector<std::string> ssb_storage_paths;
    std::vector<std::string> nvme_dev_paths;
    std::string ssb_storage_paths_raw;
    std::string nvmepaths_raw;
    std::string timelogpath;
    bool is_blksort;
    bool sorted_keys = false;
    bool profile_one_blk = false;
    bool posixapi_transfer = false;
    int sorttype; /* Deprecated */
    int sortkeyorder;
    int sort_byte_size;

    int numpartitionbucket = 1;

    // Common arguments - input
    std::string input_path; // For input file's entire path
                           // e.g., **/**/**.tbl or **/**/**.bin
    std::string input_path_1;
    std::string input_path_2;
    std::string input_path_3;
    std::string input_path_4;
    int bloomfilter_enabled; 
    int numcputhreads;
    int fascalsqlcompression; 
    int manual_selectivity;
    int ablation_level;
    int buffer_allocation_method_is_zero_copy;
    int ablationindicator;
    std::string input_dir; // For input file directory
                           // e.g., **/** (make sure that last '/' should be included)

    std::string input_name; // For input file name
                            // e.g., **.bin (Make sure that extention such as '.bin' or '.tbl' should be INCLUDED)

    // Arguments for tabulation hash join test
    std::string htb_size_str; // This is for tabulation hash join test
                              // Hash table size (e.g., 256MiB)

    std::string ptb_size_str; // This is for tabulation hash join test
                              // Probe table size (e.g., 1GiB)

    uint64_t base_profile_byte_size;

    int verbose = 0;
    bool do_save_compressed_file = false;

    /** CQO: CPU contention in [0,1]. 0=idle, 1=heavy. Drives workload_dist_ratio. */
    float cpu_contention = 0.0f;
    /** CQO: optional selectivity per predicate (for AFP pushdown decision). */
    static constexpr int kMaxCqoPredicates = 16;
    float predicate_selectivities[kMaxCqoPredicates];
    int num_predicate_selectivities = 0;

    /** TPC-H: when true, compare query result with reference (requires result output in qN.out format). */
    bool validate_tpch = false;
    /** TPC-H: directory containing reference answers q1.out .. q22.out (pipe-separated, one header line). */
    std::string reference_dir;
} arguments_t;
