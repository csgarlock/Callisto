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

# Gather source files
CU_SRCS     := $(wildcard $(SRC_DIR)/*.cu)
CPP_SRCS    := $(wildcard $(SRC_DIR)/*.cpp)

# Generate object file names in build/
CU_OBJS     := $(patsubst $(SRC_DIR)/%.cu, $(BUILD_DIR)/%.o, $(CU_SRCS))
CPP_OBJS    := $(patsubst $(SRC_DIR)/%.cpp, $(BUILD_DIR)/%.o, $(CPP_SRCS))
OBJS        := $(CU_OBJS) $(CPP_OBJS)

# Output binary
TARGET      := $(BIN_DIR)/trainer

# Default target
all: dirs $(TARGET)

# Link step
$(TARGET): $(OBJS)
	@echo "Linking $(TARGET)"
	$(NVCC) $(OBJS) -o $@ $(LDFLAGS)

# Compile CUDA source files
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cu
	@echo "Compiling CUDA: $<"
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# Compile C++ source files
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	@echo "Compiling C++: $<"
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Create directories if missing
dirs:
	@mkdir -p $(BUILD_DIR) $(BIN_DIR)

# Clean build artifacts
clean:
	@echo "Cleaning up..."
	rm -rf $(BUILD_DIR)/* $(BIN_DIR)/*

# Phony targets
.PHONY: all clean dirs
