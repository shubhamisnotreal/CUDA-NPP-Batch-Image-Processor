/*
 * CUDA at Scale Course Assignment
 * Batch Image Processing Pipeline using NVIDIA Performance Primitives (NPP)
 *
 * Target Architecture: NVIDIA T4 (Compute Capability sm_75) / CUDA 12.x
 * Google C++ Style Guide Compliant
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <npp.h>
#include <nppi.h>
#include <npps.h>

#include <opencv2/opencv.hpp>

namespace fs = std::filesystem;

// Named Constants following Google C++ Style (kConstantName)
constexpr float kDefaultSigma = 2.0f;
constexpr int kDefaultKernelSize = 9;
constexpr int kDefaultBatchSize = 16;
const std::string kDefaultFilter = "gaussian";

// Macro for CUDA Runtime API Call Verification
#define CUDA_CHECK(call)                                                \
  do {                                                                  \
    cudaError_t err = (call);                                           \
    if (err != cudaSuccess) {                                           \
      std::cerr << "CUDA Error [" << __FILE__ << ":" << __LINE__ << "]: " \
                << cudaGetErrorString(err) << " (code " << err << ")"   \
                << std::endl;                                           \
      std::exit(EXIT_FAILURE);                                          \
    }                                                                   \
  } while (0)

// Macro for NVIDIA Performance Primitives (NPP) Return Code Verification
#define NPP_CHECK(call)                                                 \
  do {                                                                  \
    NppStatus status = (call);                                          \
    if (status != NPP_SUCCESS) {                                        \
      std::cerr << "NPP Error [" << __FILE__ << ":" << __LINE__ << "]: "  \
                << "NppStatus code " << static_cast<int>(status)        \
                << std::endl;                                           \
      std::exit(EXIT_FAILURE);                                          \
    }                                                                   \
  } while (0)

// Structure holding parsed command-line interface arguments
struct AppOptions {
  std::string input_path;
  std::string output_path;
  std::string filter_type = kDefaultFilter;
  float sigma = kDefaultSigma;
  int kernel_size = kDefaultKernelSize;
  bool run_cpu = false;
  int batch_size = kDefaultBatchSize;
};

// Prints CLI usage instructions
void PrintUsage(const char* prog_name) {
  std::cout << "===============================================================\n"
            << "     CUDA NPP Batch Image Processing CLI - Usage Guide\n"
            << "===============================================================\n"
            << "Usage: " << prog_name << " [options]\n\n"
            << "Options:\n"
            << "  --input  <dir|file>   Input directory or image file (required)\n"
            << "  --output <dir>        Output directory for processed images (required)\n"
            << "  --filter <name>       Filter operation: \"gaussian\" | \"sobel\" (default: gaussian)\n"
            << "  --sigma  <float>      Gaussian sigma value (default: 2.0)\n"
            << "  --kernel <int>        Kernel mask size (odd number: 3, 5, 7, 9, 11, 13, 15; default: 9)\n"
            << "  --batch  <int>        CUDA stream batching size (default: 16)\n"
            << "  --cpu                 Run CPU baseline comparison for timing benchmark\n"
            << "  --help                Print this usage guide and exit\n"
            << "===============================================================\n";
}

// Parses command-line arguments without third-party dependencies
AppOptions ParseCommandLineOptions(int argc, char** argv) {
  AppOptions options;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--input" && i + 1 < argc) {
      options.input_path = argv[++i];
    } else if (arg == "--output" && i + 1 < argc) {
      options.output_path = argv[++i];
    } else if (arg == "--filter" && i + 1 < argc) {
      options.filter_type = argv[++i];
    } else if (arg == "--sigma" && i + 1 < argc) {
      options.sigma = std::stof(argv[++i]);
    } else if (arg == "--kernel" && i + 1 < argc) {
      options.kernel_size = std::stoi(argv[++i]);
    } else if (arg == "--batch" && i + 1 < argc) {
      options.batch_size = std::stoi(argv[++i]);
    } else if (arg == "--cpu") {
      options.run_cpu = true;
    } else if (arg == "--help") {
      PrintUsage(argv[0]);
      std::exit(EXIT_SUCCESS);
    } else {
      std::cerr << "Error: Unknown or invalid argument '" << arg << "'\n";
      PrintUsage(argv[0]);
      std::exit(EXIT_FAILURE);
    }
  }

  if (options.input_path.empty() || options.output_path.empty()) {
    std::cerr << "Error: Both --input and --output arguments must be specified.\n";
    PrintUsage(argv[0]);
    std::exit(EXIT_FAILURE);
  }

  return options;
}

// Maps integer kernel size to NPP NppiMaskSize enum
NppiMaskSize GetNppMaskSize(int kernel_size) {
  switch (kernel_size) {
    case 3:
      return NPP_MASK_SIZE_3_X_3;
    case 5:
      return NPP_MASK_SIZE_5_X_5;
    case 7:
      return NPP_MASK_SIZE_7_X_7;
    case 9:
      return NPP_MASK_SIZE_9_X_9;
    case 11:
      return NPP_MASK_SIZE_11_X_11;
    case 13:
      return NPP_MASK_SIZE_13_X_13;
    case 15:
      return NPP_MASK_SIZE_15_X_15;
    default:
      std::cout << "[!] Warning: Kernel size " << kernel_size
                << " not directly standard in NPP enum; using 9x9 mask.\n";
      return NPP_MASK_SIZE_9_X_9;
  }
}

// Collects supported image files from a directory or single file path
std::vector<fs::path> CollectImageFiles(const std::string& input_path) {
  std::vector<fs::path> image_files;
  fs::path p(input_path);

  if (!fs::exists(p)) {
    return image_files;
  }

  if (fs::is_regular_file(p)) {
    image_files.push_back(p);
  } else if (fs::is_directory(p)) {
    for (const auto& entry : fs::directory_iterator(p)) {
      if (entry.is_regular_file()) {
        std::string ext = entry.path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        if (ext == ".png" || ext == ".jpg" || ext == ".jpeg" ||
            ext == ".tif" || ext == ".tiff" || ext == ".bmp") {
          image_files.push_back(entry.path());
        }
      }
    }
  }

  std::sort(image_files.begin(), image_files.end());
  return image_files;
}

// Executes CPU baseline image processing using OpenCV
double ProcessCpuBaseline(const cv::Mat& src, cv::Mat& dst,
                         const std::string& filter_type, float sigma,
                         int kernel_size) {
  auto start_time = std::chrono::high_resolution_clock::now();

  if (filter_type == "gaussian") {
    cv::GaussianBlur(src, dst, cv::Size(kernel_size, kernel_size), sigma, sigma);
  } else if (filter_type == "sobel") {
    cv::Mat grad_x, grad_y;
    cv::Sobel(src, grad_x, CV_16S, 1, 0, 3);
    cv::Sobel(src, grad_y, CV_16S, 0, 1, 3);
    cv::Mat abs_grad_x, abs_grad_y;
    cv::convertScaleAbs(grad_x, abs_grad_x);
    cv::convertScaleAbs(grad_y, abs_grad_y);
    cv::addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, dst);
  }

  auto end_time = std::chrono::high_resolution_clock::now();
  return std::chrono::duration<double, std::milli>(end_time - start_time).count();
}

int main(int argc, char** argv) {
  AppOptions options = ParseCommandLineOptions(argc, argv);

  // Validate input path and collect dataset images
  std::vector<fs::path> image_files = CollectImageFiles(options.input_path);
  if (image_files.empty()) {
    std::cout << "===============================================================\n"
              << " NOTICE: Input path '" << options.input_path << "' is empty or contains\n"
              << " no supported image files (.png, .jpg, .tif, .bmp).\n"
              << " Please populate the directory with SIPI dataset images and retry.\n"
              << "===============================================================\n";
    return EXIT_SUCCESS;
  }

  // Ensure output directory exists
  fs::create_directories(options.output_path);

  // Query and display GPU device details
  int device_id = 0;
  CUDA_CHECK(cudaGetDevice(&device_id));
  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id));

  std::cout << "===============================================================\n"
            << "              CUDA NPP BATCH IMAGE PROCESSOR\n"
            << "===============================================================\n"
            << "GPU Device               : " << prop.name << " (Compute "
            << prop.major << "." << prop.minor << ")\n"
            << "Total Image Count        : " << image_files.size() << "\n"
            << "Filter Mode              : " << options.filter_type << "\n"
            << "Kernel Mask / Sigma      : " << options.kernel_size << "x"
            << options.kernel_size << " / " << options.sigma << "\n"
            << "Stream Batch Size        : " << options.batch_size << "\n"
            << "CPU Baseline Comparison  : " << (options.run_cpu ? "ENABLED" : "DISABLED") << "\n"
            << "===============================================================\n\n";

  NppiMaskSize npp_mask_size = GetNppMaskSize(options.kernel_size);
  size_t total_images = image_files.size();
  double total_gpu_time_ms = 0.0;
  double total_cpu_time_ms = 0.0;

  auto overall_start = std::chrono::high_resolution_clock::now();

  // Process images in batches using CUDA streams to overlap H2D, Kernel, D2H
  for (size_t batch_start = 0; batch_start < total_images; batch_start += options.batch_size) {
    size_t batch_end = std::min(batch_start + options.batch_size, total_images);
    size_t current_batch_size = batch_end - batch_start;

    std::vector<cudaStream_t> streams(current_batch_size);
    std::vector<cv::Mat> src_mats(current_batch_size);
    std::vector<cv::Mat> dst_mats(current_batch_size);
    std::vector<Npp8u*> d_src_ptrs(current_batch_size, nullptr);
    std::vector<Npp8u*> d_dst_ptrs(current_batch_size, nullptr);
    std::vector<size_t> bytes_list(current_batch_size, 0);

    // Create CUDA streams for batch
    for (size_t i = 0; i < current_batch_size; ++i) {
      CUDA_CHECK(cudaStreamCreate(&streams[i]));
    }

    // Step 1: Load images and enqueue GPU pipeline per stream
    for (size_t i = 0; i < current_batch_size; ++i) {
      size_t global_idx = batch_start + i;
      const auto& file_path = image_files[global_idx];

      std::cout << "[" << (global_idx + 1) << "/" << total_images << "] Processing "
                << file_path.filename().string() << std::endl;

      // Load image from host disk
      cv::Mat img = cv::imread(file_path.string(), cv::IMREAD_UNCHANGED);
      if (img.empty()) {
        std::cerr << "Warning: Failed to load image " << file_path << ". Skipping.\n";
        continue;
      }

      // Convert 4-channel RGBA/BGRA to 3-channel BGR
      if (img.channels() == 4) {
        cv::cvtColor(img, img, cv::COLOR_BGRA2BGR);
      }

      src_mats[i] = img;
      dst_mats[i] = cv::Mat(img.rows, img.cols, img.type());

      size_t img_bytes = img.rows * img.step;
      bytes_list[i] = img_bytes;

      // Register pinned CPU memory for high-throughput asynchronous transfer
      CUDA_CHECK(cudaHostRegister(src_mats[i].data, img_bytes, cudaHostRegisterPortable));
      CUDA_CHECK(cudaHostRegister(dst_mats[i].data, img_bytes, cudaHostRegisterPortable));

      // Allocate GPU device memory
      CUDA_CHECK(cudaMalloc(&d_src_ptrs[i], img_bytes));
      CUDA_CHECK(cudaMalloc(&d_dst_ptrs[i], img_bytes));

      // Asynchronous Host-to-Device memory transfer
      CUDA_CHECK(cudaMemcpyAsync(d_src_ptrs[i], src_mats[i].data, img_bytes,
                                 cudaMemcpyHostToDevice, streams[i]));

      // Configure NPP stream context
      NppStreamContext npp_ctx;
      NPP_CHECK(nppGetStreamContext(&npp_ctx));
      npp_ctx.hStream = streams[i];

      NppiSize roi_size = {img.cols, img.rows};
      int src_step = static_cast<int>(img.step);
      int dst_step = static_cast<int>(dst_mats[i].step);
      int channels = img.channels();

      // Launch NPP Filter kernel on stream
      if (options.filter_type == "gaussian") {
        if (channels == 1) {
          NPP_CHECK(nppiFilterGaussian_8u_C1R_Ctx(
              d_src_ptrs[i], src_step, d_dst_ptrs[i], dst_step, roi_size,
              npp_mask_size, npp_ctx));
        } else if (channels == 3) {
          NPP_CHECK(nppiFilterGaussian_8u_C3R_Ctx(
              d_src_ptrs[i], src_step, d_dst_ptrs[i], dst_step, roi_size,
              npp_mask_size, npp_ctx));
        } else {
          std::cerr << "Unsupported channel count: " << channels << std::endl;
        }
      } else if (options.filter_type == "sobel") {
        if (channels == 1) {
          NPP_CHECK(nppiFilterSobelVert_8u_C1R_Ctx(
              d_src_ptrs[i], src_step, d_dst_ptrs[i], dst_step, roi_size,
              npp_mask_size, npp_ctx));
        } else if (channels == 3) {
          NPP_CHECK(nppiFilterSobelVert_8u_C3R_Ctx(
              d_src_ptrs[i], src_step, d_dst_ptrs[i], dst_step, roi_size,
              npp_mask_size, npp_ctx));
        } else {
          std::cerr << "Unsupported channel count: " << channels << std::endl;
        }
      }

      // Asynchronous Device-to-Host memory transfer
      CUDA_CHECK(cudaMemcpyAsync(dst_mats[i].data, d_dst_ptrs[i], img_bytes,
                                 cudaMemcpyDeviceToHost, streams[i]));

      // Optional CPU baseline timing execution
      if (options.run_cpu) {
        cv::Mat cpu_dst;
        double cpu_ms = ProcessCpuBaseline(src_mats[i], cpu_dst,
                                          options.filter_type, options.sigma,
                                          options.kernel_size);
        total_cpu_time_ms += cpu_ms;
      }
    }

    // Synchronize batch streams, write output files, and cleanup host/device memory
    for (size_t i = 0; i < current_batch_size; ++i) {
      if (streams[i] != nullptr) {
        CUDA_CHECK(cudaStreamSynchronize(streams[i]));

        // Write processed frame to disk
        size_t global_idx = batch_start + i;
        fs::path out_file_path = fs::path(options.output_path) / image_files[global_idx].filename();
        cv::imwrite(out_file_path.string(), dst_mats[i]);

        // Unregister host memory and free device pointers
        CUDA_CHECK(cudaHostUnregister(src_mats[i].data));
        CUDA_CHECK(cudaHostUnregister(dst_mats[i].data));
        CUDA_CHECK(cudaFree(d_src_ptrs[i]));
        CUDA_CHECK(cudaFree(d_dst_ptrs[i]));
        CUDA_CHECK(cudaStreamDestroy(streams[i]));
      }
    }
  }

  auto overall_end = std::chrono::high_resolution_clock::now();
  total_gpu_time_ms = std::chrono::duration<double, std::milli>(overall_end - overall_start).count();

  // Print execution summary report
  std::cout << "\n===============================================================\n"
            << "              EXECUTION & TIMING SUMMARY REPORT\n"
            << "===============================================================\n"
            << "Total Images Processed   : " << total_images << "\n"
            << "Total GPU Time           : " << std::fixed << std::setprecision(2)
            << total_gpu_time_ms << " ms\n"
            << "Average GPU Time / Image : " << (total_gpu_time_ms / total_images) << " ms\n";

  if (options.run_cpu) {
    std::cout << "Total CPU Baseline Time  : " << total_cpu_time_ms << " ms\n"
              << "Average CPU Time / Image : " << (total_cpu_time_ms / total_images) << " ms\n"
              << "Speedup Factor (CPU/GPU) : " << (total_cpu_time_ms / total_gpu_time_ms) << "x\n";
  }
  std::cout << "===============================================================\n";

  return EXIT_SUCCESS;
}
