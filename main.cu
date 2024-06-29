#include <cuda.h>
#include <cuda_runtime.h>

#include <array>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
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
  const uint32_t stepCount = 2;

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
  std::cout << "Result: ";
  for (uint8_t c : result) {
    std::cout << (uint32_t)c;
  }
  std::cout << std::endl;
}

int main(int p_argc, char *p_argv[]) {
  const uint32_t repeatCount = 10000;

  if (p_argc < 2) {
    std::cout << "Usage: aoc-2019-16 <signal file | raw signal>";
    return 0;
  }
  NC_ASSERT_CUDA(cuInit(0));
  CUdevice dev;
  int devCount;
  NC_ASSERT_CUDA(cuDeviceGetCount(&devCount));
  if (devCount == 0) {
    std::cerr << "No CUDA devices available." << std::endl;
    return 1;
  }
  NC_ASSERT_CUDA(cuDeviceGet(&dev, 0));
  char devName[128];
  NC_ASSERT_CUDA(cuDeviceGetName(devName, sizeof(devName), dev));
  std::cout << "Selected CUDA device: " << devName << std::endl;
  CUcontext ctx;
  NC_ASSERT_CUDA(cuCtxCreate(&ctx, 0, dev));

  std::string input;
  std::ifstream in(p_argv[1], std::ios::ate);
  if (in) {
    input.resize(in.tellg());
    in.seekg(0);
    in.read(input.data(), input.size());
  } else {
    input = p_argv[1];
  }
  for (char c : input) {
    if (c < '0' || '9' < c) {
      std::cerr << "Every single character in the signal string must be digit between 0 and 9, inclusively."
                << std::endl;
      return 1;
    }
  }
  if (input.size() < 7) {
    std::cerr << "Signal length must be at least 7." << std::endl;
    return 1;
  }
  uint32_t resultOffset = 0;
  for (uint32_t i = 0; i < 7; ++i) {
    resultOffset = 10 * resultOffset + (input[i] - '0');
  }
  if (repeatCount * input.size() <= resultOffset + 8) {
    std::cerr << "First 7 digits of the signal interpreted as a single number + 8 must be less than or equal to the "
                 "repeated signal's length."
              << std::endl;
    return 1;
  }

  std::vector<uint8_t> signal(repeatCount * input.size());
  for (size_t i = 0; i < signal.size(); ++i) {
    signal[i] = input[i % input.size()] - '0';
  }
  run(signal, resultOffset);
  return 0;
}