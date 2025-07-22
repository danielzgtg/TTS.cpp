add_library(parler
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
add_subdirectory(t5)
target_link_libraries(parler PRIVATE tts-base)
target_link_libraries(tts PRIVATE parler)
