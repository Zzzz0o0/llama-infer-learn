cmake_minimum_required(VERSION 3.16)

project(llama_infer CXX CUDA)

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)

aux_source_directory(kuiper/source/base DIR_BASE)
aux_source_directory(kuiper/source/tensor DIR_TENSOR)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

add_library(llama SHARED ${DIR_BASE} ${DIR_TENSOR})

target_include_directories(llama PUBLIC ${PROJECT_SOURCE_DIR}/kuiper/include)

add_subdirectory(test)