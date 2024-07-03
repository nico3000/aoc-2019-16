#include <cuda.h>
#include <cuda_runtime.h>

#include <array>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#define NC_ASSERT_CUDA(_cmd)                                                                                           \
  do {                                                                                                                 \
    if (CUresult _res = _cmd; _res != CUDA_SUCCESS) {                                                                  \
      const char *error = "unknown cuda error";                                                                        \
      cuGetErrorName(_res, &error);                                                                                    \
      std::cerr << error << std::endl;                                                                                 \
      exit(1);                                                                                                         \
    }                                                                                                                  \
  } while (0)

__global__ void fft(const uint8_t *p_srcSignal, uint8_t *p_dstSignal, uint32_t p_signalSize) {
  uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  printf("test\n");
  if (p_signalSize <= idx) {
    return;
  }
  uint32_t i = idx;
  int32_t sum = 0;
  int32_t f = 1;
  while (i < p_signalSize) {
    uint32_t partialSum = 0;
    for (uint32_t j = 0; j <= idx && i < p_signalSize; ++j, ++i) {
      partialSum += p_srcSignal[i];
    }
    sum += f * (int32_t)partialSum;
    f = -f;
    i += idx + 1;
  }
  p_dstSignal[idx] = (sum < 0 ? -sum : sum) % 10;
}

void run(const std::vector<uint8_t> &p_signal, uint32_t p_resultOffset) {
  const uint32_t stepCount = 100;

  CUdevice dev;
  NC_ASSERT_CUDA(cuDeviceGet(&dev, 0));
  char devName[128];
  NC_ASSERT_CUDA(cuDeviceGetName(devName, sizeof(devName), dev));
  std::cout << "Selected CUDA device: " << devName << std::endl;

  CUcontext ctx;
  NC_ASSERT_CUDA(cuCtxCreate(&ctx, CU_CTX_SCHED_BLOCKING_SYNC, dev));
  NC_ASSERT_CUDA(cuCtxPushCurrent(ctx));

  CUstream stream;
  NC_ASSERT_CUDA(cuStreamCreate(&stream, CU_STREAM_DEFAULT));

  CUdeviceptr srcSignalDevMem, dstSignalDevMem;
  NC_ASSERT_CUDA(cuMemAlloc(&srcSignalDevMem, p_signal.size()));
  NC_ASSERT_CUDA(cuMemAlloc(&dstSignalDevMem, p_signal.size()));
  NC_ASSERT_CUDA(cuMemcpyHtoD(srcSignalDevMem, p_signal.data(), p_signal.size()));

  std::cout << "Ready for number crunching!" << std::endl;
  for (uint32_t i = 0; i < stepCount; ++i) {
    auto beg = std::chrono::high_resolution_clock::now();
    uint32_t numThreadsPerBlock = 128;
    uint32_t numBlocks = (p_signal.size() + numThreadsPerBlock - 1) / numThreadsPerBlock;
    fft<<<numBlocks, numThreadsPerBlock, 0, stream>>>((const uint8_t *)srcSignalDevMem, (uint8_t *)dstSignalDevMem,
                                                      (uint32_t)p_signal.size());
    NC_ASSERT_CUDA(cuStreamSynchronize(stream));
    std::swap(srcSignalDevMem, dstSignalDevMem);
    auto end = std::chrono::high_resolution_clock::now();
    auto seconds = std::chrono::duration_cast<std::chrono::duration<float>>(end - beg);
    std::cout << "Step " << (i + 1) << " / " << stepCount << " done. (" << std::fixed << std::setprecision(3)
              << seconds.count() << " s)" << std::endl;
  }

  std::array<uint8_t, 8> result;
  NC_ASSERT_CUDA(cuMemcpyDtoH(result.data(), srcSignalDevMem + p_resultOffset, result.size()));
  NC_ASSERT_CUDA(cuMemFree(dstSignalDevMem));
  NC_ASSERT_CUDA(cuMemFree(srcSignalDevMem));
  NC_ASSERT_CUDA(cuStreamDestroy(stream));
  NC_ASSERT_CUDA(cuCtxPopCurrent(nullptr));
  NC_ASSERT_CUDA(cuCtxDestroy(ctx));

  std::cout << "Result: ";
  for (uint8_t c : result) {
    std::cout << (uint32_t)c;
  }
  std::cout << std::endl;
}

void printUsage() {
  std::cout << "Usage:" << std::endl
            << "  aoc-2019-16 [-p <nr> | --part <nr>] (<raw_signal> | <signal_file_path>)" << std::endl
            << "  aoc-2019-16 -h | --help" << std::endl
            << "Compute the solution of Advent of Code 2019, day 16 for a given signal (the puzzle input) with CUDA."
            << std::endl
            << std::endl
            << "Options:" << std::endl
            << "  -h --help\t\tPrint this screen." << std::endl
            << "  -p <nr> --part=<nr>\tSelect the part of the puzzle. Either 1 or 2. [default: 2]" << std::endl;
}

int main(int p_argc, char *p_argv[]) {
  if (p_argc < 2) {
    std::cerr << "Missing signal. Call with --help to print usage information." << std::endl;
    return 1;
  }

  if (!strcmp(p_argv[1], "--help") || !strcmp(p_argv[1], "-h")) {
    printUsage();
    return 0;
  }

  NC_ASSERT_CUDA(cuInit(0));
  int devCount;
  NC_ASSERT_CUDA(cuDeviceGetCount(&devCount));
  if (devCount == 0) {
    std::cerr << "No CUDA devices available." << std::endl;
    return 1;
  }

  uint32_t part = 2;
  uint32_t signalArgIdx = 1;
  if (!strcmp(p_argv[1], "-p") || !strcmp(p_argv[1], "--part")) {
    if (p_argc < 3) {
      std::cerr << "Missing part number. Must be 1 or 2." << std::endl;
      return 1;
    } else if (!strcmp(p_argv[2], "1")) {
      part = 1;
    } else if (strcmp(p_argv[2], "2")) {
      std::cerr << "Invalid part number. Must be 1 or 2, is " << p_argv[2] << "." << std::endl;
      return 1;
    }
    signalArgIdx += 2;
  }
  if (signalArgIdx + 1 < p_argc) {
    std::cerr << "Too many arguments." << std::endl;
    return 1;
  } else if (p_argc <= signalArgIdx) {
    std::cerr << "Missing signal argument." << std::endl;
    return 1;
  }

  std::string input;
  std::ifstream in(p_argv[signalArgIdx], std::ios::ate);
  if (in) {
    input.resize(in.tellg());
    in.seekg(0);
    in.read(input.data(), input.size());
  } else {
    input = p_argv[signalArgIdx];
  }
  for (char c : input) {
    if (c < '0' || '9' < c) {
      std::cerr << "Every single character in the signal must be a digit between 0 and 9, inclusively." << std::endl;
      return 1;
    }
  }

  uint32_t resultOffset = 0;
  const uint32_t repeatCount = part == 1 ? 1 : 10000;
  if (part == 2) {
    if (input.size() < 7) {
      std::cerr << "Signal length fort part 2 must be at least 7." << std::endl;
      return 1;
    }
    for (uint32_t i = 0; i < 7; ++i) {
      resultOffset = 10 * resultOffset + (input[i] - '0');
    }
    if (repeatCount * input.size() <= resultOffset + 8) {
      std::cerr << "First 7 digits of the signal interpreted as a single number + 8 must be less than or equal to the "
                   "repeated signal's length."
                << std::endl;
      return 1;
    }
  }

  std::vector<uint8_t> signal(repeatCount * input.size());
  for (size_t i = 0; i < signal.size(); ++i) {
    signal[i] = input[i % input.size()] - '0';
  }

  run(signal, resultOffset);
  return 0;
}
