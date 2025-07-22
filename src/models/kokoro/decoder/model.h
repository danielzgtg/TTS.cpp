#pragma once

#include "buffers.h"
#include "imports.h"

struct ggml_tensor;
struct gguf_context;

struct kokoro_generator_residual_block {
    vector<uint32_t> conv1_dilations;
    vector<uint32_t> conv1_paddings;

    vector<ggml_tensor *> adain1d_1_gamma_weights;
    vector<ggml_tensor *> adain1d_2_gamma_weights;
    vector<ggml_tensor *> adain1d_1_gamma_biases;
    vector<ggml_tensor *> adain1d_2_gamma_biases;
    vector<ggml_tensor *> adain1d_1_beta_weights;
    vector<ggml_tensor *> adain1d_2_beta_weights;
    vector<ggml_tensor *> adain1d_1_beta_biases;
    vector<ggml_tensor *> adain1d_2_beta_biases;
    vector<ggml_tensor *> input_alphas;
    vector<ggml_tensor *> output_alphas;
    vector<ggml_tensor *> convs1_weights;
    vector<ggml_tensor *> convs1_biases;
    vector<ggml_tensor *> convs2_weights;
    vector<ggml_tensor *> convs2_biases;
};

struct kokoro_noise_residual_block {
    uint32_t input_conv_stride;
    uint32_t input_conv_padding;

    ggml_tensor *                     input_conv;
    ggml_tensor *                     input_conv_bias;
    kokoro_generator_residual_block * res_block;
};

struct kokoro_generator_upsample_block {
    kokoro_generator_upsample_block(gguf_context * meta, int index);
    uint32_t padding;
    uint32_t stride;

    // these are just conv transpose layers
    ggml_tensor * upsample_weight{};
    ggml_tensor * upsample_bias{};
};

struct kokoro_generator {
    // unfortunately the squared sum of the windows needs to be computed dynamically per run because it is dependent
    // on the sequence size of the generation and the hop is typically less than half the size of our window.
    ggml_tensor * window;

    ggml_tensor *                             m_source_weight;
    ggml_tensor *                             m_source_bias;
    ggml_tensor *                             out_conv_weight;
    ggml_tensor *                             out_conv_bias;
    vector<kokoro_noise_residual_block *>     noise_blocks;
    vector<kokoro_generator_residual_block *> res_blocks;
    vector<kokoro_generator_upsample_block *> ups;
};

struct kokoro_decoder {
    ggml_tensor *                     f0_conv;
    ggml_tensor *                     f0_conv_bias;
    ggml_tensor *                     n_conv;
    ggml_tensor *                     n_conv_bias;
    ggml_tensor *                     asr_conv;
    ggml_tensor *                     asr_conv_bias;
    vector<ada_residual_conv_block *> decoder_blocks;
    ada_residual_conv_block *         encoder_block;
    kokoro_generator *                generator;
};
