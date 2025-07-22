#pragma once

struct parler_kv_cache {
    int32_t seq_id;

    ggml_type type_k = GGML_TYPE_F32;
    ggml_type type_v = GGML_TYPE_F32;

    std::vector<ggml_tensor *> k_l;
    std::vector<ggml_tensor *> v_l;

    struct ggml_context *      ctx;
    ggml_backend_buffer_type_t buft;
    ggml_backend_buffer_t      buf;

    ~parler_kv_cache() {
        ggml_free(ctx);
        ggml_backend_buffer_free(buf);
    }
};
