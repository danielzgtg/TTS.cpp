add_library(phonemizer
        ../../../include/phonemizer.h
        phonemizer.cpp
)
target_include_directories(phonemizer PUBLIC ../../../include)
target_link_libraries(phonemizer PUBLIC tokenizer)
if (ESPEAK_INCLUDE_DIRS)
    set_source_files_properties(phonemizer.cpp PROPERTIES COMPILE_FLAGS "${ESPEAK_CFLAGS_OTHER}")
    set_source_files_properties(phonemizer.cpp PROPERTIES INCLUDE_DIRECTORIES "${ESPEAK_INCLUDE_DIRS}")
    target_link_libraries(phonemizer PRIVATE ${ESPEAK_LIBRARIES})
endif ()

add_library(kokoro
        build.cpp
        context.cpp
        context.h
        hparams.cpp
        loader.cpp
        model.h
        tensors.cpp
)
add_subdirectory(albert)
add_subdirectory(decoder)
add_subdirectory(duration)
add_subdirectory(lstm)
target_link_libraries(kokoro PRIVATE tts-base phonemizer)
target_link_libraries(tts PRIVATE kokoro)
