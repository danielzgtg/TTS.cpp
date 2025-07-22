add_library(dia
        build.cpp
        context.cpp
        context.h
        hparams.cpp
        kv.cpp
        kv.h
        loader.cpp
        model.h
        tensors.cpp
        tokenize.cpp
)
add_subdirectory(decoder)
add_subdirectory(encoder)
target_link_libraries(dia PRIVATE tts-base)
target_link_libraries(tts PRIVATE dia)
