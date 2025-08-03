
add_executable(warp_reduce warp_reduce.cu)
target_link_libraries(
        warp_reduce PUBLIC
        -lcudart
        -lcudadevrt)
