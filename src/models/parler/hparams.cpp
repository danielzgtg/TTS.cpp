
void parler_tts_model::prep_layers(gguf_context * meta_ctx) {
    layers.reserve((size_t) n_layers);
    for (int i = 0; i < (int) n_layers; i++) {
        parler_layer * l = new parler_layer{};
        layers.push_back(l);
    }

    embds.reserve((size_t) n_output_heads);
    heads.reserve((size_t) n_output_heads);
    for (int i = 0; i < n_output_heads; i++) {
        struct ggml_tensor * h = nullptr;
        struct ggml_tensor * embd = nullptr;
        embds.push_back(embd);
        heads.push_back(h);
    }
}

void parler_tts_model::prep_constants(gguf_context * meta) {
    int encode_length_key = search_for_gguf_keys(meta, {"parler-tts.decoder.encode_length", "encode_length"});
    if (encode_length_key == -1) {
        TTS_ABORT("key 'parler-tts.decoder.encode_length' must be specified in gguf file.");
    }
    n_encode_length = gguf_get_val_u32(meta, encode_length_key);

    int hidden_size_key = search_for_gguf_keys(meta, {"parler-tts.decoder.hidden_size", "hidden_size"});
    if (hidden_size_key != -1) {
        hidden_size = gguf_get_val_u32(meta, hidden_size_key);
    }

    int output_heads_key = search_for_gguf_keys(meta, {"parler-tts.decoder.output_heads", "output_heads"});
    if (output_heads_key != -1) {
        n_output_heads = gguf_get_val_u32(meta, output_heads_key);
    }
    int ctx_length_key = search_for_gguf_keys(meta, {"parler-tts.decoder.context_length", "ctx_length"});
    if (ctx_length_key != -1) {
        max_ctx_length = gguf_get_val_u32(meta, ctx_length_key);
    }

    int attn_heads_key = search_for_gguf_keys(meta, {"parler-tts.decoder.attention.head_count", "attn_heads"});
    if (attn_heads_key != -1) {
        n_attn_heads = gguf_get_val_u32(meta, attn_heads_key);
    }
    head_size = hidden_size / n_attn_heads;
    max_cross_nodes = n_attn_heads * 2;

    int output_vocab_size_key = search_for_gguf_keys(meta, {"parler-tts.decoder.out_vocab_size", "out_vocab_size"});
    if (output_vocab_size_key != -1) {
        output_vocab_size = gguf_get_val_u32(meta, output_vocab_size_key);
    }

    int audio_vocab_size_key = search_for_gguf_keys(meta, {"parler-tts.decoder.audio_vocab_size", "audio_vocab_size"});
    if (audio_vocab_size_key != -1) {
        audio_vocab_size = gguf_get_val_u32(meta, audio_vocab_size_key);
    }

    int max_gen_key = search_for_gguf_keys(meta, {"parler-tts.decoder.max_generation", "max_generation"});
    if (max_gen_key != -1) {
        max_generation_size = gguf_get_val_u32(meta, max_gen_key);
    }

    int n_layers_key = search_for_gguf_keys(meta, {"parler-tts.decoder.num_hidden_layers", "num_hidden_layers"});
    if (n_layers_key != -1) {
        n_layers = gguf_get_val_u32(meta, n_layers_key);
    }

    int bos_token_id_key = search_for_gguf_keys(meta, {"audio.bos_token_id", "bos_token_id"});
    if (bos_token_id_key != -1) {
        bos_token_id = gguf_get_val_u32(meta, bos_token_id_key);
    }

    int eos_token_id_key = search_for_gguf_keys(meta, {"audio.eos_token_id", "eos_token_id"});
    if (eos_token_id_key != -1) {
        eos_token_id = gguf_get_val_u32(meta, eos_token_id_key);
    }
}
