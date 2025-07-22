#pragma once

struct ggml_tensor;

struct albert_layer {
    ggml_tensor * ffn;
    ggml_tensor * ffn_out;
    ggml_tensor * ffn_bias;
    ggml_tensor * ffn_out_bias;
    ggml_tensor * layer_output_norm_weight;
    ggml_tensor * layer_output_norm_bias;
    ggml_tensor * q;
    ggml_tensor * k;
    ggml_tensor * v;
    ggml_tensor * o;
    ggml_tensor * q_bias;
    ggml_tensor * k_bias;
    ggml_tensor * v_bias;
    ggml_tensor * o_bias;
    ggml_tensor * attn_norm_weight;
    ggml_tensor * attn_norm_bias;
};
