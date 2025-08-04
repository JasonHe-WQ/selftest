
add_executable(gemm gemm.cu)
target_link_libraries(
        gemm PUBLIC
        -lcudart
        -lcudadevrt)
