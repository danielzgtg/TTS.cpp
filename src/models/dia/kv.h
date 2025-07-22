#pragma once

struct dia_kv_cache {
    ggml_type tensor_type = GGML_TYPE_F32;

    vector<ggml_tensor *> cross_k_l;
    vector<ggml_tensor *> cross_v_l;

    vector<ggml_tensor *> k_l;
    vector<ggml_tensor *> v_l;

    ggml_context * ctx;
    ggml_backend_buffer_type_t buft;
    ggml_backend_buffer_t buf;

    ~dia_kv_cache() {
        ggml_free(ctx);
        ggml_backend_buffer_free(buf);
    }
};
