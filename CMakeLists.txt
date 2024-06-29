cmake_minimum_required(VERSION 3.28)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)

find_package(CUDAToolkit REQUIRED)

project(aoc-2019-16 LANGUAGES CXX CUDA)

add_executable(aoc-2019-16 main.cu)
target_link_libraries(aoc-2019-16 CUDA::cuda_driver CUDA::cudart)
