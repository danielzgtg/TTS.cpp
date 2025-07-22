#pragma once

#include <cstddef>

// TODO ggml-cpp
struct ggml_backend;
struct ggml_backend_buffer;
struct ggml_context;
struct ggml_tensor;
struct ggml_threadpool;

/**
 * Like VkDeviceMemory.
 * However, neither this NOR the underlying resources are thread-safe.
 */
class tts_backend_memory {
    explicit tts_backend_memory(bool cpu_only);
    ~tts_backend_memory();
    ggml_backend *    _gpu{};
    ggml_backend *    cpu{};
    ggml_threadpool * threadpool;
};

/// Like VkBuffer
class tts_buffer {
    /// this is the current byte offset into the model's buffer.
    size_t                offset{}; // TODO let either ggml_tallocr or ggml_backend.cpp do this?
    ggml_backend_buffer * buf{};
    ggml_context *        ctx{};
    friend struct tts_model;
  public:
    explicit tts_buffer(int n_threads);
    ~tts_buffer();
    void set_tensor(ggml_tensor * tensor, ggml_tensor * target);

    // ReSharper disable once CppNonExplicitConversionOperator // We want this to automatically cast
    operator ggml_context *() const {  // NOLINT(*-explicit-constructor)
        return ctx;
    }
};
