add_library(tokenizer
        ../include/args_config.h
        ../include/imports.h
        ../include/tokenizer.h
        tokenizer.cpp
        util.h # TODO move to tts-base once string operations are replaced by C++20
        util.cpp
)
target_include_directories(tokenizer PUBLIC ../include)
target_include_directories(tokenizer PUBLIC ../ggml/src) # TODO rm once string operations are replaced by C++20
target_link_libraries(tokenizer PUBLIC ggml)

add_library(tts-base
        ../include/common.h
        buffers.cpp
        buffers.h
        sampler.cpp
        sampler.h
        tts_model.cpp
)
add_subdirectory(decoder)
target_include_directories(tts-base PUBLIC . ../ggml/src ./models)
target_link_libraries(tts-base PUBLIC tokenizer)

add_subdirectory(models)
