
add_executable(histogram histogram.cu)
target_link_libraries(
        histogram PUBLIC
        -lcudart
        -lcudadevrt)
