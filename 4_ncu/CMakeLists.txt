
add_executable(combine_access combine_access.cu)
target_link_libraries(
        combine_access PUBLIC
        -lcudart
        -lcudadevrt)
target_compile_options(combine_access PRIVATE -lineinfo)