#!/usr/bin/env bash
# CUDA NPP Batch Image Processing Pipeline Driver Script
# Target: Ubuntu Colab with NVIDIA T4 GPU

set -e

DATA_DIR="./data"
OUTPUT_DIR="./output"
BIN="./bin/processor"
LOG_FILE="$OUTPUT_DIR/log.txt"

echo "==============================================================="
echo "        CUDA NPP Batch Image Processing Pipeline"
echo "==============================================================="

# Step 1: Ensure executable binary exists (compile if missing)
if [ ! -f "$BIN" ]; then
    echo "[*] Executable '$BIN' not found. Compiling codebase..."
    make all
fi

# Step 2: Verify ./data contains input image files
if [ ! -d "$DATA_DIR" ] || [ -z "$(ls -A "$DATA_DIR" 2>/dev/null)" ]; then
    echo ""
    echo "---------------------------------------------------------------"
    echo " ERROR: Input dataset directory '$DATA_DIR' is missing or empty!"
    echo " Please populate '$DATA_DIR' with USC SIPI dataset images"
    echo " (PNG, TIFF, JPG, BMP) before executing run.sh."
    echo ""
    echo " Example command to download sample dataset:"
    echo "   mkdir -p data && wget -P data http://sipi.usc.edu/database/download.php?vol=misc"
    echo "---------------------------------------------------------------"
    exit 1
fi

# Step 3: Create output directory
mkdir -p "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR/sobel"

# Initialize execution logging
echo "[*] Execution started at: $(date)" | tee "$LOG_FILE"
echo "[*] Input Dataset Directory: $DATA_DIR" | tee -a "$LOG_FILE"
echo "[*] Output Directory       : $OUTPUT_DIR" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Step 4: Run Batch Gaussian Filter with CPU Baseline Comparison
echo ">>> [1/2] Running GPU vs CPU Benchmark (Gaussian Filter, Kernel 9x9, Sigma 2.0)..." | tee -a "$LOG_FILE"
"$BIN" --input "$DATA_DIR" --output "$OUTPUT_DIR" --filter gaussian --sigma 2.0 --kernel 9 --cpu --batch 16 2>&1 | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"

# Step 5: Run Batch Sobel Edge Detection
echo ">>> [2/2] Running GPU Batch Sobel Edge Detection..." | tee -a "$LOG_FILE"
"$BIN" --input "$DATA_DIR" --output "$OUTPUT_DIR/sobel" --filter sobel --kernel 3 --batch 16 2>&1 | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo "===============================================================" | tee -a "$LOG_FILE"
echo " PIPELINE EXECUTION COMPLETED SUCCESSFULLY!" | tee -a "$LOG_FILE"
echo " - Log File Saved To  : $LOG_FILE" | tee -a "$LOG_FILE"
echo " - Processed Outputs  : $OUTPUT_DIR" | tee -a "$LOG_FILE"
echo "===============================================================" | tee -a "$LOG_FILE"
