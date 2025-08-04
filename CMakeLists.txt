cmake_minimum_required(VERSION 3.20 FATAL_ERROR)
project(selftest LANGUAGES CXX CUDA)

find_package(CUDAToolkit )
set(CUDA_PATH ${CUDA_TOOLKIT_ROOT_DIR})
set(CMAKE_C_FLAGS    "${CMAKE_C_FLAGS}")
set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS}")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}  -Xcompiler -Wall")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}  \
                        -gencode=arch=compute_70,code=\\\"sm_70,compute_70\\\" \
                        -gencode=arch=compute_75,code=\\\"sm_75,compute_75\\\" \
                        -gencode=arch=compute_80,code=\\\"sm_80,compute_80\\\" \
                        -gencode=arch=compute_86,code=\\\"sm_86,compute_86\\\" \
                        ")
set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -g -G")
set(COMMON_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}
        ${CUDA_PATH}/include
)

set(COMMON_LIB_DIRS
        ${CUDA_PATH}/lib64
)
include_directories(
        ${COMMON_HEADER_DIRS}
)
link_directories(
        ${COMMON_LIB_DIRS}
)

add_subdirectory(1_reduce)
add_subdirectory(2_histogram)
add_subdirectory(3_gemm)