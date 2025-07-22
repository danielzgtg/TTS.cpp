#pragma once

struct t5_layer {
    ggml_tensor * q;
    ggml_tensor * k;
    ggml_tensor * v;
    ggml_tensor * o;
    ggml_tensor * attn_norm;
    ggml_tensor * wi_0;
    ggml_tensor * wi_1;
    ggml_tensor * wo;
    ggml_tensor * mlp_norm;
};

// this struct maintains the static tensors for a t5_encoder model
// the default configuration is form copied from standard configuration for
// flan-t5-xl. Note this model is slightly different from a standard T5 encoder.
// Specifically this model has a down projection which converts the text encoder's
// hidden size to the hidden size of the parler decoder.
struct t5_encoder : tts_model {
    // These configs  are essentially built for the 44khZ 8kbps standard DAC model audio encoder and decoder
    uint32_t n_layers = 24;
    uint32_t n_attn_heads = 32;
    uint32_t head_size = 64;
    uint32_t hidden_size = 2048;
    uint32_t relative_attn_buckets = 32;
    uint32_t eos_token_id = 1;
    uint32_t bos_token_id = 0;
    uint32_t max_context_length = 512;
    uint32_t output_size = 1536;
    uint32_t vocab_size;

    ggml_tensor * embd;
    ggml_tensor * relative_attn_bias;
    ggml_tensor * out_norm;
    ggml_tensor * down_proj = nullptr;
    ggml_tensor * down_proj_bias = nullptr;
    vector<t5_layer> layers;

    void assign_weight(std::string name, ggml_tensor * tensor);
    void prep_layers(gguf_context * meta);
    void prep_constants(gguf_context * meta);
    void setup_from_file(gguf_context * meta_ctx, ggml_context * load_context, bool cpu_only = true) {
        prep_constants(meta_ctx);
        prep_layers(meta_ctx);
        tts_model::setup_from_file(load_context, cpu_only, "t5encoder", 1.25);
    }
};
