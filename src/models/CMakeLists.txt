target_sources(tts-base PRIVATE
        loaders.h
)
target_include_directories(tts-base PUBLIC .)

add_library(tts
        ../../include/tts.h
        ../tts.cpp # TODO merge and delete
        loaders.cpp
)
target_link_libraries(tts PUBLIC tts-base)

add_library(quantize-base
        quantizers.cpp
        quantizers.h
        quantizers_impl.h
)
target_link_libraries(quantize-base PUBLIC ggml)
target_include_directories(quantize-base PUBLIC . ../../include)

foreach (_ARCH IN ITEMS # TODO STOPSHIP: uncomment
#        DIA
#        KOKORO
        PARLER
)
    option(TTS_ARCH_${_ARCH} "" ON)
    if (TTS_ARCH_${_ARCH})
        string(TOLOWER ${_ARCH} _ARCH)
        message(STATUS "Enabling TTS model architecture: ${_ARCH}")
        add_subdirectory(${_ARCH})
        target_sources(quantize-base PRIVATE ${_ARCH}/quantizer.cpp)
        # If models are not in individual .so files, use this to ensure [[constructor]]s are called
        set(LD_WORKAROUND "${LD_WORKAROUND}void ${_ARCH}_register();${_ARCH}_register();")
    endif ()
endforeach ()

set_source_files_properties(loaders.cpp PROPERTIES COMPILE_FLAGS "-DLD_WORKAROUND=\"${LD_WORKAROUND}\"")
set_source_files_properties(quantizers.cpp PROPERTIES COMPILE_FLAGS "-DLD_WORKAROUND=\"${LD_WORKAROUND}\"")
