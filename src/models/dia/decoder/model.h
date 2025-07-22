#pragma once

#include "imports.h"

struct dia_decoder_layer {
    ggml_tensor * self_attn_k;
    ggml_tensor * self_attn_q;
    ggml_tensor * self_attn_v;
    ggml_tensor * self_attn_o;
    ggml_tensor * self_attn_norm;
    
    ggml_tensor * cross_attn_k;
    ggml_tensor * cross_attn_q;
    ggml_tensor * cross_attn_v;
    ggml_tensor * cross_attn_o;
    ggml_tensor * cross_attn_norm;

    ggml_tensor * gate;
    ggml_tensor * up;
    ggml_tensor * out;
    ggml_tensor * mlp_norm;

    ggml_tensor * pad_attn_values;
};

struct dia_decoder {
    ggml_tensor * norm;
    vector<ggml_tensor*> embds;
    vector<ggml_tensor*> heads;
    vector<dia_decoder_layer*> layers;
};
