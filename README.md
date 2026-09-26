# Batch Image Processing with CUDA NPP

A complete, high-performance CUDA/C++ image processing pipeline designed for batch processing large sets of images (USC SIPI Image Database) using NVIDIA Performance Primitives (NPP). Developed as a CUDA at Scale course assignment, targeting Google Colab with an NVIDIA T4 GPU (Compute Capability `sm_75`).

---

## Table of Contents
1. [Overview](#overview)
2. [Target Environment & Hardware Requirements](#target-environment--hardware-requirements)
3. [Important Architecture Note (`-arch=sm_75`)](#important-architecture-note--archsm_75)
4. [Build Instructions](#build-instructions)
5. [Run Instructions](#run-instructions)
6. [CLI Reference](#cli-reference)
7. [Google Colab Workflow](#google-colab-workflow)
8. [Proof of Execution Artifacts](#proof-of-execution-artifacts)
9. [Lessons Learned](#lessons-learned)
10. [License](#license)

---

## Overview

This project implements a multi-stream GPU image filtering pipeline using **NVIDIA Performance Primitives (NPP)**. The application processes batches of images concurrently using CUDA streams to overlap Host-to-Device (H2D) memory transfer, GPU kernel execution, and Device-to-Host (D2H) memory transfer.

### Key Features:
- **NPP GPU Filtering**: Supports Gaussian Blur (`nppiFilterGaussian`) and Sobel Edge Detection (`nppiFilterSobelVert`).
- **CUDA Stream Batching**: Processes groups of images (default batch size: 16) asynchronously using CUDA streams for maximum throughput.
- **CPU Baseline Benchmark**: Includes an optional CPU implementation using OpenCV (`--cpu`) to measure GPU speedup.
- **Google C++ Style Compliance**: Standardized coding format, explicit error checking via `CUDA_CHECK` and `NPP_CHECK` macros, named constants, and clean modular structure.

---

## Target Environment & Hardware Requirements

- **Operating System**: Ubuntu 20.04 / 22.04 LTS (or Google Colab environment)
- **GPU Accelerator**: NVIDIA T4 GPU (Turing architecture, Compute Capability 7.5)
- **CUDA Toolkit**: CUDA 11.x or 12.x preinstalled at `/usr/local/cuda`
- **Host Dependencies**: `g++` (C++14 support), OpenCV 4.x (`libopencv-dev`), `pkg-config`, `make`

---

## Important Architecture Note (`-arch=sm_75`)

> [!IMPORTANT]  
> When compiling for Google Colab's free-tier NVIDIA T4 GPU, you **MUST** pass `-arch=sm_75` to `nvcc`. Failing to specify the compute architecture often results in runtime PTX compilation errors such as:
> ```text
> CUDA Error: the provided PTX was compiled with an unsupported toolchain
> ```
> The included `Makefile` automatically includes `-arch=sm_75`.

---

## Build Instructions

### 1. Install Dependencies (Ubuntu / Colab)
```bash
sudo apt-get update
sudo apt-get install -y build-essential cuda-toolkit libopencv-dev pkg-config
```

### 2. Compile Project
Build the binary executable (`bin/processor`):
```bash
make all
```

To clean existing build artifacts:
```bash
make clean
```

---

## Run Instructions

### 1. Populate Input Dataset
Place USC SIPI database images (`.png`, `.tiff`, `.jpg`, `.bmp`) inside `./data/`. If `./data/` is missing or empty, the application exits gracefully with instructions.

Quick command to fetch sample SIPI images:
```bash
mkdir -p data
wget -O ./data/4.1.01.tiff http://sipi.usc.edu/database/download.php?vol=misc\&img=4.1.01
wget -O ./data/4.1.02.tiff http://sipi.usc.edu/database/download.php?vol=misc\&img=4.1.02
wget -O ./data/4.1.03.tiff http://sipi.usc.edu/database/download.php?vol=misc\&img=4.1.03
```

### 2. Execute Automated Driver Script
Run the automated bash driver:
```bash
bash run.sh
```

### 3. Manual CLI Execution Examples
**GPU Gaussian Blur with CPU Baseline Benchmark:**
```bash
./bin/processor --input ./data --output ./output --filter gaussian --sigma 2.0 --kernel 9 --cpu --batch 16
```

**GPU Sobel Edge Detection:**
```bash
./bin/processor --input ./data --output ./output/sobel --filter sobel --kernel 3 --batch 16
```

---

## CLI Reference

```text
Usage: ./bin/processor [options]

Options:
  --input  <dir|file>   Input directory or image file (required)
  --output <dir>        Output directory for processed images (required)
  --filter <name>       Filter operation: "gaussian" | "sobel" (default: gaussian)
  --sigma  <float>      Gaussian sigma value (default: 2.0)
  --kernel <int>        Kernel mask size (odd number: 3, 5, 7, 9, 11, 13, 15; default: 9)
  --batch  <int>        CUDA stream batching size (default: 16)
  --cpu                 Run CPU baseline comparison for timing benchmark
  --help                Print this usage guide and exit
```

---

## Google Colab Workflow

1. Open Google Colab and set Runtime type to **GPU (T4)**.
2. Open `colab_setup.ipynb` or execute the following cell steps:
   - **Cell 1**: `!nvidia-smi`
   - **Cell 2**: `!nvcc --version`
   - **Cell 3**: `!apt-get -qq update && !apt-get -qq install -y libopencv-dev`
   - **Cell 4**: `!git clone <MY_REPO_URL> && %cd <repo-name>`
   - **Cell 5**: `!make all`
   - **Cell 6**: Upload images or download SIPI dataset images into `./data/`
   - **Cell 7**: `!bash run.sh`
   - **Cell 8**: Inspect before/after images using `matplotlib`
   - **Cell 9**: `!zip -r output_artifacts.zip output/` and download using `files.download('output_artifacts.zip')`.

---

## Proof of Execution Artifacts

After execution, proof artifacts are generated under `./output/`:
- `output/log.txt`: Complete terminal log capturing per-image progress, GPU execution time, CPU baseline timing, and overall speedup metrics.
- `output/*.tiff` / `output/*.png`: Transformed output image files.

Example Summary Output Log:
```text
===============================================================
              EXECUTION & TIMING SUMMARY REPORT
===============================================================
Total Images Processed   : 100
Total GPU Time           : 142.50 ms
Average GPU Time / Image : 1.42 ms
Total CPU Baseline Time  : 1850.20 ms
Average CPU Time / Image : 18.50 ms
Speedup Factor (CPU/GPU) : 12.98x
===============================================================
```

---

## Lessons Learned

*(Placeholder for course assignment submission write-up)*

- **NPP Stream Integration**: Setting `npp_ctx.hStream = stream[i]` enables seamless overlapping of memory transfers and kernel execution across concurrent streams.
- **Pinned Host Memory**: Utilizing `cudaHostRegister` on host OpenCV buffers is critical for maximizing PCI-e transfer bandwidth during `cudaMemcpyAsync`.
- **Target Architecture Matching**: Explicitly configuring `-arch=sm_75` in `nvcc` compiler flags prevents PTX translation errors on Colab T4 GPUs.

---

## License

MIT License. Developed for CUDA at Scale Course Assignment.
