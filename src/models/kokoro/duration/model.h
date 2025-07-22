#pragma once

#include "ada/model.h"
#include "../lstm/model.h"

struct duration_predictor_layer {
    lstm *        rnn;
    ggml_tensor * ada_norm_gamma_weight;
    ggml_tensor * ada_norm_gamma_bias;
    ggml_tensor * ada_norm_beta_weight;
    ggml_tensor * ada_norm_beta_bias;
};

struct duration_predictor {
    ggml_tensor *                      albert_encode;
    ggml_tensor *                      albert_encode_bias;
    vector<duration_predictor_layer *> layers;
    lstm *                             duration_proj_lstm;
    ggml_tensor *                      duration_proj;
    ggml_tensor *                      duration_proj_bias;
    ggml_tensor *                      n_proj_kernel;
    ggml_tensor *                      n_proj_bias;
    ggml_tensor *                      f0_proj_kernel;
    ggml_tensor *                      f0_proj_bias;
    lstm *                             shared_lstm;
    vector<ada_residual_conv_block *>  f0_blocks;
    vector<ada_residual_conv_block *>  n_blocks;
};
