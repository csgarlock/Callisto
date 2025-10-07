# ===== Makefile for CUDA + C++ =====

# Compiler settings
NVCC        := nvcc
CXX         := g++
CXXFLAGS    := -O3 -std=c++17
NVCCFLAGS   := -O3 -std=c++17 -arch=sm_89
LDFLAGS     := -lcudart

# Directory paths
SRC_DIR     := src
BUILD_DIR   := build
BIN_DIR     := bin

# Gather all source files recursively
CU_SRCS     := $(shell find $(SRC_DIR) -type f -name '*.cu')
CPP_SRCS    := $(shell find $(SRC_DIR) -type f -name '*.cpp')

# Map src/... to build/... (preserve directory structure)
CU_OBJS     := $(patsubst $(SRC_DIR)/%.cu,$(BUILD_DIR)/%.o,$(CU_SRCS))
CPP_OBJS    := $(patsubst $(SRC_DIR)/%.cpp,$(BUILD_DIR)/%.o,$(CPP_SRCS))
OBJS        := $(CU_OBJS) $(CPP_OBJS)

# Output binary
TARGET      := $(BIN_DIR)/trainer

# Default target
all: dirs $(TARGET)

# Link step
$(TARGET): $(OBJS)
	@echo "Linking $(TARGET)"
	$(NVCC) $(OBJS) -o $@ $(LDFLAGS)

# Pattern rule for CUDA compilation
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cu
	@mkdir -p $(dir $@)
	@echo "Compiling CUDA: $<"
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# Pattern rule for C++ compilation
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(dir $@)
	@echo "Compiling C++: $<"
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Create output directories
dirs:
	@mkdir -p $(BUILD_DIR) $(BIN_DIR)

# Clean build artifacts
clean:
	@echo "Cleaning up..."
	rm -rf $(BUILD_DIR) $(BIN_DIR)

# Phony targets
.PHONY: all clean dirs
