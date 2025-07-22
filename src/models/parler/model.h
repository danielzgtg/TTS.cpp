#pragma once

struct parler_layer {
    ggml_tensor * self_attn_k_proj;
    ggml_tensor * self_attn_q_proj;
    ggml_tensor * self_attn_v_proj;
    ggml_tensor * self_attn_o_proj;
    ggml_tensor * self_attn_norm;
    ggml_tensor * self_attn_norm_bias;

    ggml_tensor * attn_k_proj;
    ggml_tensor * attn_q_proj;
    ggml_tensor * attn_v_proj;
    ggml_tensor * attn_o_proj;
    ggml_tensor * attn_norm;
    ggml_tensor * attn_norm_bias;

    ggml_tensor * cross_k;
    ggml_tensor * cross_v;

    ggml_tensor * fc1;
    ggml_tensor * fc2;
    ggml_tensor * final_norm;
    ggml_tensor * final_norm_bias;
};

struct parler_tts_model : tts_model {
    // These default configurations are based on the configuration of Parler TTS Mini (version 1.0)
    uint32_t n_output_heads = 9;
    uint32_t n_encode_length;
    uint32_t max_encode_length   = 512;  // This corresponds with the max token length of the conditional prompt
    uint32_t hidden_size         = 1024;
    uint32_t max_ctx_length      = 4096;
    uint32_t n_attn_heads        = 16;
    uint32_t head_size           = 64;
    uint32_t output_vocab_size   = 1088;
    uint32_t eos_token_id        = 1024;
    uint32_t audio_vocab_size    = 1024;
    uint32_t max_generation_size = 2580;
    uint32_t n_layers            = 24;
    uint32_t bos_token_id        = 1025;
    uint32_t max_cross_nodes     = 32;
    uint32_t prompt_vocab_size;

    bool use_cross_attn = true;

    vector<ggml_tensor *>  embds;
    vector<parler_layer *> layers;
    vector<ggml_tensor *>  heads;

    ggml_tensor * precomputed_input_emb;
    ggml_tensor * precomputed_positional_embds;

    ggml_tensor * layer_norm;
    ggml_tensor * layer_norm_bias;
    ggml_tensor * prompt_embd;

    void assign_weight(const char * name, ggml_tensor * tensor);
    void prep_constants(gguf_context * meta);
    void prep_layers(gguf_context * meta);
    void prep_cross_key_values(int n_threads, struct tts_response * conditional_prompt = nullptr);

    void setup_from_file(gguf_context * meta_ctx, ggml_context * load_context, bool cpu_only) {
        prep_constants(meta_ctx);
        prep_layers(meta_ctx);
        tts_model::setup_from_file(load_context, cpu_only, "decoder", 1.30,
                                   max_encode_length * hidden_size * sizeof(float) * n_layers * 2);
    }
};
