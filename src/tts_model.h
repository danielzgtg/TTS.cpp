#pragma once

#include <cstring>
#include <functional>

#include "backend_manager.h"
#include "common.h"
#include "util.h"

void append_to_response(tts_response * response, const tts_response * to_append);

using tensor_meta_callback = std::function<void(ggml_tensor *)> *;

struct runner_context {
    runner_context(int n_threads) : n_threads(n_threads) {};

    virtual ~runner_context() {
        ggml_backend_sched_free(sched);
        ggml_threadpool_free(threadpool);
        ggml_backend_buffer_free(buf_output);
    }


    std::vector<uint8_t>  buf_compute_meta;
    ggml_backend_buffer_t buf_output = nullptr;
    ggml_backend_sched_t  sched      = nullptr;
    ggml_threadpool_t     threadpool = nullptr;
    int                   n_threads;

    void get_ggml_node_data(struct ggml_tensor * output_tensor, float * output, size_t output_size,
                            ggml_backend_buffer_t buffer = nullptr) const;
    void set_threads();
    void build_schedule(size_t max_nodes);
    bool prep_schedule(ggml_cgraph * gf) const;
};

struct tts_model {
    model_tensor_meta tensor_meta;

    bool use_cross_attn = true;

    ggml_backend_buffer_type_t buffer  = nullptr;
    ggml_backend_t             backend = nullptr;

    // it is quite common for implementations of tts_model to need to update attributes or perform distinct operations
    // when computing the tensor meta of the loaded model. This callback allows this as it will receive each processed tensor.
    tensor_meta_callback compute_tensor_meta_cb = nullptr;

    backend_context backends; // TODO move buffer and backend here

    void         prep_buffers_and_context(bool cpu_only, float size_offset, uint32_t dedicated_add_on_size);
    void         setup_from_file(ggml_context * load_context, bool cpu_only, const string & model_prefix,
                                 float size_offset = 1.4, uint32_t dedicated_add_on_size = 0);
    size_t       max_nodes() const;
    virtual ~tts_model();
};
