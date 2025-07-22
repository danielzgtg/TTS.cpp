#pragma once

#include "imports.h"

struct dia_encoder_layer {
    ggml_tensor * k;
    ggml_tensor * q;
    ggml_tensor * v;
    ggml_tensor * o;
    ggml_tensor * self_attn_norm;

    ggml_tensor * gate;
    ggml_tensor * up;
    ggml_tensor * out;
    ggml_tensor * mlp_norm;
};

struct dia_encoder {
    ggml_tensor *               norm;
    ggml_tensor *               embedding;
    vector<dia_encoder_layer *> layers;
};
