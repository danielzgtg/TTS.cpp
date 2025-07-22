
void dia_model::prep_constants(gguf_context * meta) {
    int output_heads_key = gguf_find_key(meta, "dia.decoder.output_heads");
    if (output_heads_key != -1) {
        n_output_heads = gguf_get_val_u32(meta, output_heads_key);
    }

    int decoder_layers_key = gguf_find_key(meta, "dia.decoder.layers");
    if (decoder_layers_key != -1) {
        n_decoder_layers = gguf_get_val_u32(meta, decoder_layers_key);
    }

    int encoder_layers_key = gguf_find_key(meta, "dia.encoder.layers");
    if (encoder_layers_key != -1) {
        n_encoder_layers = gguf_get_val_u32(meta, encoder_layers_key);
    }

    int decoder_hidden_size_key = gguf_find_key(meta, "dia.decoder.hidden_size");
    if (decoder_hidden_size_key != -1) {
        decoder_hidden_size = gguf_get_val_u32(meta, decoder_hidden_size_key);
    }

    int decoder_attn_heads_key = gguf_find_key(meta, "dia.decoder.attn_heads");
    if (decoder_attn_heads_key != -1) {
        decoder_attn_heads = gguf_get_val_u32(meta, decoder_attn_heads_key);
    }

    int decoder_query_heads_key = gguf_find_key(meta, "dia.decoder.query_heads");
    if (decoder_query_heads_key != -1) {
        decoder_query_heads = gguf_get_val_u32(meta, decoder_query_heads_key);
    }

    int encoder_attn_heads_key = gguf_find_key(meta, "dia.encoder.attn_heads");
    if (encoder_attn_heads_key != -1) {
        encoder_attn_heads = gguf_get_val_u32(meta, encoder_attn_heads_key);
    }

    int head_size_key = gguf_find_key(meta, "dia.attn_head_size");
    if (head_size_key != -1) {
        head_size = gguf_get_val_u32(meta, head_size_key);
    }

    int eos_token_id_key = gguf_find_key(meta, "dia.eos_token_id");
    if (eos_token_id_key != -1) {
        eos_token_id = gguf_get_val_u32(meta, eos_token_id_key);
    }

    int bos_token_id_key = gguf_find_key(meta, "dia.bos_token_id");
    if (bos_token_id_key != -1) {
        bos_token_id = gguf_get_val_u32(meta, bos_token_id_key);
    }

    int pad_token_id_key = gguf_find_key(meta, "dia.pad_token_id");
    if (pad_token_id_key != -1) {
        pad_token_id = gguf_get_val_u32(meta, pad_token_id_key);
    }

    int max_context_key = gguf_find_key(meta, "dia.encoder.max_context_length");
    if (max_context_key != -1) {
        max_encoder_context_length = gguf_get_val_u32(meta, max_context_key);
    }

    int output_vocab_size_key = gguf_find_key(meta, "dia.decoder.output_vocab_size");
    if (output_vocab_size_key != -1) {
        output_vocab_size = gguf_get_val_u32(meta, output_vocab_size_key);
    }

    int audio_vocab_size_key = gguf_find_key(meta, "dia.decoder.audio_vocab_size");
    if (audio_vocab_size_key != -1) {
        audio_vocab_size = gguf_get_val_u32(meta, audio_vocab_size_key);
    }

    int max_generation_size_key = gguf_find_key(meta, "dia.decoder.max_generation_size");
    if (max_generation_size_key != -1) {
        max_generation_size = gguf_get_val_u32(meta, max_generation_size_key);
    }
    int max_delay_key = gguf_find_key(meta, "dia.max_delay");
    if (max_delay_key != -1) {
        max_delay = gguf_get_val_u32(meta, max_delay_key);
    }

    // please note that this value is not currently set in the gguf encoder as it effectively only exists as a default
    // python parameter (rather than an attribute in the model config) for the python Dia model.
    int cfg_scale_key = gguf_find_key(meta, "dia.cfg_scale");
    if (cfg_scale_key != -1) {
        cfg_scale_data[0] = gguf_get_val_f32(meta, cfg_scale_key);
    }
}

void dia_model::prep_layers() {
    encoder = new dia_encoder;
    decoder = new dia_decoder;
    encoder->layers.reserve((size_t) n_encoder_layers);
    for (int i = 0; i < (int) n_encoder_layers; i++) {
        dia_encoder_layer * l = new dia_encoder_layer;
        encoder->layers.push_back(l);
    }

    decoder->layers.reserve((size_t) n_decoder_layers);
    for (int i = 0; i < (int) n_decoder_layers; i++) {
        dia_decoder_layer * l = new dia_decoder_layer;
        decoder->layers.push_back(l);
    }

    decoder->embds.reserve((size_t) n_output_heads);
    decoder->heads.reserve((size_t) n_output_heads);
    for (int i = 0; i < n_output_heads; i++) {
        ggml_tensor * h    = nullptr;
        ggml_tensor * embd = nullptr;
        decoder->embds.push_back(embd);
        decoder->heads.push_back(h);
    }
}
