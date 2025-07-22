add_executable(phonemize phonemize.cpp)
target_link_libraries(phonemize PRIVATE examples-base phonemizer)
target_link_options(phonemize PRIVATE -Wl,--gc-sections)
