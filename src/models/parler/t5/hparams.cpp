
void t5_encoder::prep_layers(gguf_context * meta) {
    for (uint32_t i = 0; i < n_layers; i++) {
        t5_layer l;
        layers.push_back(l);
    }
}

void t5_encoder::prep_constants(gguf_context * meta) {
    int n_layers_key = gguf_find_key(meta, "t5encoder.block_count");
    if (n_layers_key != -1) {
        n_layers = gguf_get_val_u32(meta, n_layers_key);
    }

    int hidden_size_key = gguf_find_key(meta, "t5encoder.embedding_length");
    if (hidden_size_key != -1) {
        hidden_size = gguf_get_val_u32(meta, hidden_size_key);
    }

    int attn_heads_key = gguf_find_key(meta, "t5encoder.attention.head_count");
    if (attn_heads_key != -1) {
        n_attn_heads = gguf_get_val_u32(meta, attn_heads_key);
    }

    int context_size_key = gguf_find_key(meta, "t5encoder.context_length");
    if (context_size_key != -1) {
        max_context_length = gguf_get_val_u32(meta, context_size_key);
    }

    int bos_token_id_key = gguf_find_key(meta, "tokenizer.ggml.bos_token_id");
    if (bos_token_id_key != -1) {
        bos_token_id = gguf_get_val_u32(meta, bos_token_id_key);
    }

    int eos_token_id_key = gguf_find_key(meta, "tokenizer.ggml.eos_token_id");
    if (eos_token_id_key != -1) {
        eos_token_id = gguf_get_val_u32(meta, eos_token_id_key);
    }

    int vocab_size_key = gguf_find_key(meta, "t5encoder.vocab_size");
    if (vocab_size_key == -1) {
        TTS_ABORT("key 't5encoder.vocab_size' must be specified in gguf file.");
    }
    vocab_size = gguf_get_val_u32(meta, vocab_size_key);

    int output_size_key = gguf_find_key(meta, "t5encoder.output_size");
    if (output_size_key != -1) {
        output_size = gguf_get_val_u32(meta, output_size_key);
    }
}
