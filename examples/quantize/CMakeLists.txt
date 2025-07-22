add_executable(quantize
        quantize.cpp
        quantize_impl.cpp
        quantize_impl.h
)
target_link_libraries(quantize PRIVATE examples-base quantize-base)
#target_compile_options(quantize PRIVATE -ftime-trace)
