#pragma once

#include "imports.h"

class tts_buffer;
struct ggml_context;
struct ggml_tensor;

struct ada_residual_conv_block {
    ggml_tensor * conv1{};
    ggml_tensor * conv1_bias{};
    ggml_tensor * conv2{};
    ggml_tensor * conv2_bias{};
    ggml_tensor * norm1_gamma{};
    ggml_tensor * norm1_gamma_bias{};
    ggml_tensor * norm1_beta{};
    ggml_tensor * norm1_beta_bias{};
    ggml_tensor * norm2_gamma{};
    ggml_tensor * norm2_gamma_bias{};
    ggml_tensor * norm2_beta{};
    ggml_tensor * norm2_beta_bias{};
    ggml_tensor * pool{};
    ggml_tensor * pool_bias{};
    ggml_tensor * upsample{};
    ggml_tensor * upsample_bias{};

    void          assign_ada_res_block(tts_buffer & ctx, string_view name, ggml_tensor * tensor);
    ggml_tensor * build(ggml_context * ctx, ggml_tensor * x, ggml_tensor * style, ggml_tensor * sqrt_tensor) const;
};
