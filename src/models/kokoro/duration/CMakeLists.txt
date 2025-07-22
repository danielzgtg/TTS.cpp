target_sources(kokoro PRIVATE
        build.cpp
        context.cpp
        context.h
        model.h
        tensors.cpp
)
add_subdirectory(ada)
