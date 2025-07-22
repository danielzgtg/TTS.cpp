#pragma once

#include "tts_model.h"

struct dac_residual_unit {
    ggml_tensor * in_snake_alpha;
    ggml_tensor * in_conv_kernel;
    ggml_tensor * in_conv_bias;
    ggml_tensor * out_snake_alpha;
    ggml_tensor * out_conv_kernel;
    ggml_tensor * out_conv_bias;
};

struct dac_layer {
    ggml_tensor * snake_alpha_in;
    ggml_tensor * out_conv_kernel;
    ggml_tensor * out_conv_bias;

    uint32_t padding;
    uint32_t stride;

    vector<dac_residual_unit> residual_blocks;
};

struct dac_quantize_layer {
    ggml_tensor * out_proj_kernel;
    ggml_tensor * out_proj_bias;
    ggml_tensor * codebook;
};

struct dac_hparams {
    // These configs  are essentially built for the 44khZ 8kbps standard DAC model audio encoder and decoder
    uint32_t n_layers            = 4;
    uint32_t n_heads             = 9;
    uint32_t up_sampling_factor  = 512;
    uint32_t max_generation_size = 2580;
};

// this struct maintains the static tensors for the dac audio decoder graph.
// As such, this is designed to contain basic configuration and ggml tensor support for DAC.
// The dac_runner describes how the graph is built and run.
struct dac_model : tts_model {
    dac_hparams hparams;

    ggml_tensor *              in_conv_kernel;
    ggml_tensor *              in_conv_bias;
    ggml_tensor *              out_conv_kernel;
    ggml_tensor *              out_conv_bias;
    ggml_tensor *              snake_alpha;
    vector<dac_layer>          layers;
    vector<dac_quantize_layer> quantizer_layers;

    void assign_weight(string name, ggml_tensor * weight);
    void prep_constants(gguf_context * meta);
    void prep_layers();

    void setup_from_file(gguf_context * meta_ctx, ggml_context * load_context, bool cpu_only) {
        prep_layers();
        prep_constants(meta_ctx);
        tts_model::setup_from_file(load_context, cpu_only, "audio_encoder");
    }
};
