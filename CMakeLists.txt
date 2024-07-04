cmake_minimum_required(VERSION 3.28)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)

project(aoc-2019-16 LANGUAGES CXX CUDA)
find_package(CUDAToolkit REQUIRED)

add_executable(aoc-2019-16 main.cu)

install(TARGETS aoc-2019-16)
