#pragma once

struct dia_model : tts_model {
    // These default configurations are based on the default configuration for the Dia 1.68b param model.
    uint32_t n_output_heads = 9;
    uint32_t n_encoder_layers = 12;
    uint32_t n_decoder_layers = 18;
    uint32_t encoder_hidden_size = 1024;
    uint32_t decoder_hidden_size = 2048;
    uint32_t encoder_attn_heads = 16;
    uint32_t decoder_attn_heads = 16;
    uint32_t decoder_query_heads = 4;
    uint32_t head_size = 128;
    uint32_t eos_token_id = 1024;
    uint32_t pad_token_id = 1025;
    uint32_t bos_token_id = 1026;
    uint32_t output_vocab_size = 1028;
    uint32_t audio_vocab_size = 1024;
    uint32_t max_generation_size = 3072;
    uint32_t max_encoder_context_length = 1024;


    float cfg_scale_data[2] = {3.0, 1024.0};
    uint32_t max_delay = 15;
    std::vector<uint32_t> delay_pattern = {0, 8, 9, 10, 11, 12, 13, 14, 15};

    dia_encoder * encoder;
    dia_decoder * decoder;
    
    void assign_weight(std::string name, ggml_tensor * tensor);
    void prep_constants(gguf_context * meta);
    void prep_layers();
    void setup_from_file(gguf_context * meta_ctx, ggml_context * load_context, bool cpu_only) {
        prep_constants(meta_ctx);
        prep_layers();
        tts_model::setup_from_file(meta_ctx, load_context, cpu_only, "dia", 1.30);
    }
};
