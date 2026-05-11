# FaScalSQL Makefile
# GPU-accelerated SQL query engine — standalone binaries only

SHELL := /bin/bash

CUDA_PATH ?= /usr/local/cuda
NVCC = $(CUDA_PATH)/bin/nvcc
CXX = g++
CUDA_LDFLAGS = -L$(CUDA_PATH)/lib64

# Architecture
# Auto-detect GPU architecture if not specified by user
ifndef GPU_ARCH
  _GPU_CC_RAW := $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1)
  _GPU_CC := $(shell echo "$(_GPU_CC_RAW)" | sed -n 's/^\([0-9][0-9]*\)\.\([0-9][0-9]*\)$$/\1\2/p')
  ifneq ($(strip $(_GPU_CC)),)
    GPU_ARCH := sm_$(_GPU_CC)
  else
    GPU_ARCH := sm_86
  endif
endif

# Directories
SRC_DIR = src
INCLUDE_DIR = include
BUILD_DIR = build
REPO_ROOT = ../..
KERNELS_DIR = ../kernels

# Compiler flags
NVCC_FLAGS = -std=c++14 -arch=$(GPU_ARCH) -O3 -lineinfo --use_fast_math
CXX_FLAGS = -std=c++14 -O3 -pthread -MMD -MP -mavx2

# Standalone NVCC flags (for .cu that include AVX2 intrinsics in host code)
STANDALONE_NVCC_FLAGS = -std=c++14 -arch=$(GPU_ARCH) -O3 -lineinfo --use_fast_math \
    -Xcompiler "-mavx2 -pthread -O3"

# Includes
INCLUDES = -I$(INCLUDE_DIR) -I$(INCLUDE_DIR)/crystal -I$(CUDA_PATH)/include

# Libraries
CUDA_LIBS = -lcudart -lcuda

# ============================================================================
# Runtime library objects (ODZC, AFP, common)
# ============================================================================
RUNTIME_LIB_OBJS = $(BUILD_DIR)/odzc.o $(BUILD_DIR)/afp.o $(BUILD_DIR)/fascal_generated_common.o

.PHONY: all clean help tpch_standalone ssb_standalone standalone_all
.SECONDEXPANSION:

-include $(wildcard $(BUILD_DIR)/*.d)
-include $(wildcard $(BUILD_DIR)/standalone/*.d)

all: standalone_all

# ============================================================================
# Runtime objects (CPU code compiled with g++, GPU code with nvcc)
# ============================================================================

$(BUILD_DIR)/odzc.o: $(SRC_DIR)/odzc.cpp | $(BUILD_DIR)
	$(CXX) $(CXX_FLAGS) $(INCLUDES) -c $< -o $@

$(BUILD_DIR)/afp.o: $(SRC_DIR)/afp.cpp | $(BUILD_DIR)
	$(CXX) $(CXX_FLAGS) $(INCLUDES) -c $< -o $@

$(BUILD_DIR)/fascal_generated_common.o: $(SRC_DIR)/fascal_generated_common.cu | $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -c $< -o $@

# ============================================================================
# Standalone query binaries from .cu kernel files
# ============================================================================
# Usage:  make build/standalone/tpch_q6
#         make build/standalone/ssb_q11

# Step 1: compile .cu → .o
$(BUILD_DIR)/standalone/%.o: $(KERNELS_DIR)/%.cu $(RUNTIME_LIB_OBJS) | $(BUILD_DIR)/standalone
	$(NVCC) $(STANDALONE_NVCC_FLAGS) $(INCLUDES) -c $< -o $@

# Step 2: link .o + runtime libs → binary
$(BUILD_DIR)/standalone/%: $(BUILD_DIR)/standalone/%.o $(RUNTIME_LIB_OBJS) | $(BUILD_DIR)/standalone
	$(NVCC) $(STANDALONE_NVCC_FLAGS) $(BUILD_DIR)/standalone/$*.o $(RUNTIME_LIB_OBJS) -o $@ $(CUDA_LDFLAGS) $(CUDA_LIBS)

# ============================================================================
# Convenience targets: build all TPC-H / SSB standalone binaries
# ============================================================================

TPCH_QUERIES := $(patsubst $(KERNELS_DIR)/%.cu,%,$(wildcard $(KERNELS_DIR)/tpch_q*.cu))
SSB_QUERIES  := $(patsubst $(KERNELS_DIR)/%.cu,%,$(wildcard $(KERNELS_DIR)/ssb_q*.cu))

TPCH_BINS := $(addprefix $(BUILD_DIR)/standalone/,$(TPCH_QUERIES))
SSB_BINS  := $(addprefix $(BUILD_DIR)/standalone/,$(SSB_QUERIES))

tpch_standalone: $(TPCH_BINS)
ssb_standalone: $(SSB_BINS)
standalone_all: $(TPCH_BINS) $(SSB_BINS)

# ============================================================================
# Directory targets
# ============================================================================

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/standalone: | $(BUILD_DIR)
	mkdir -p $(BUILD_DIR)/standalone

# ============================================================================
# Clean
# ============================================================================
clean:
	rm -rf $(BUILD_DIR)

# ============================================================================
# Help
# ============================================================================
help:
	@echo "FaScalSQL Build System"
	@echo "======================="
	@echo ""
	@echo "Targets:"
	@echo "  all                       - Build all standalone binaries (default)"
	@echo "  build/standalone/<query>  - Build a standalone query binary"
	@echo "                              e.g. make build/standalone/tpch_q6"
	@echo "  tpch_standalone           - Build all 22 TPC-H standalone binaries"
	@echo "  ssb_standalone            - Build all 13 SSB standalone binaries"
	@echo "  standalone_all            - Build all 35 standalone binaries"
	@echo "  clean                     - Remove build artifacts"
	@echo ""
	@echo "Variables:"
	@echo "  CUDA_PATH - CUDA installation path (default: /usr/local/cuda)"
	@echo "  GPU_ARCH  - GPU architecture (default: auto-detect or sm_86)"
	@echo ""
	@echo "Examples:"
	@echo "  make standalone_all -j\$$(nproc)"
	@echo "  make build/standalone/tpch_q6"
	@echo "  make GPU_ARCH=sm_80 standalone_all"
