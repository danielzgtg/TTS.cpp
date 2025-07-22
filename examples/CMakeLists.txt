# examples

add_library(examples-base
        args.cpp
        args.h
)
target_link_libraries(examples-base PUBLIC ggml)
target_include_directories(examples-base PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/../include)

add_library(examples-common
        args_common.cpp
        args_common.h
        audio_file.h
)
target_link_libraries(examples-common PUBLIC examples-base tts)

if (NOT EMSCRIPTEN)
    add_subdirectory(cli)
    add_subdirectory(perf_battery)
    if (TTS_ARCH_KOKORO)
        add_subdirectory(phonemize)
    endif ()
    add_subdirectory(quantize)
    add_subdirectory(server)
endif ()
