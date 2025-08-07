
add_executable(gemm1 gemm1.cu)
add_executable(gemm2 gemm2.cu)
add_executable(gemm3 gemm3.cu)
target_link_libraries(
        gemm1 PUBLIC
        -lcudart
        -lcudadevrt)

target_link_libraries(
        gemm2 PUBLIC
        -lcudart
        -lcudadevrt
)

target_link_libraries(
        gemm3 PUBLIC
        -lcudart
        -lcudadevrt
)