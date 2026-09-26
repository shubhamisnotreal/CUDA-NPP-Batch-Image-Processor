# Makefile for CUDA NPP Batch Image Processor
# Target Architecture: Ubuntu Colab / NVIDIA T4 GPU (Compute Capability sm_75)

TARGET = bin/processor
NVCC = nvcc

# IMPORTANT COLAB NOTE:
# Colab's free T4 GPU uses compute capability sm_75.
# If compiling without -arch=sm_75, you may encounter the runtime error:
# "CUDA Error: the provided PTX was compiled with an unsupported toolchain".
CUDA_ARCH ?= -arch=sm_75
NVCCFLAGS = -std=c++14 -O2 $(CUDA_ARCH)

# NVIDIA Performance Primitives (NPP) & CUDA Linker Libraries
NPP_LIBS = -lcudart -lnppc -lnppial -lnppif -lnppig -lnppicc -lnppidei -lnppim -lnppist -lnppisu -lnppitc -lculibos

# OpenCV Configuration via pkg-config (fallback to opencv if opencv4 not explicit)
OPENCV_CFLAGS := $(shell pkg-config --cflags opencv4 2>/dev/null || pkg-config --cflags opencv 2>/dev/null)
OPENCV_LIBS := $(shell pkg-config --libs opencv4 2>/dev/null || pkg-config --libs opencv 2>/dev/null)

INCLUDES = $(OPENCV_CFLAGS)
LIBS = $(NPP_LIBS) $(OPENCV_LIBS)

SRCS = src/main.cu

.PHONY: all clean run dirs help

all: dirs $(TARGET)

dirs:
	@mkdir -p bin output data

$(TARGET): $(SRCS)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $< -o $@ $(LIBS)

clean:
	rm -rf bin/ output/ *.o

run: all
	./bin/processor --input ./data --output ./output --filter gaussian --sigma 2.0 --kernel 9 --cpu

help:
	@echo "Available Makefile Targets:"
	@echo "  make       - Compile the CUDA NPP processor binary (bin/processor)"
	@echo "  make clean - Remove compiled binary and output files"
	@echo "  make run   - Execute the batch processing pipeline on ./data"
