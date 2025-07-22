
void dia_model::assign_to_decoder_layer(string part, dia_decoder_layer * layer, ggml_tensor * tensor) {
    if (part == "self_q_proj") {
        layer->self_attn_q = ggml_dup_tensor(ctx, tensor);
        set_tensor(layer->self_attn_q, tensor);
    } else if (part == "self_k_proj") {
        layer->self_attn_k = ggml_dup_tensor(ctx, tensor);
        set_tensor(layer->self_attn_k, tensor);
    } else if (part == "self_v_proj") {
        layer->self_attn_v = ggml_dup_tensor(ctx, tensor);
        set_tensor(layer->self_attn_v, tensor);
    } else if (part == "self_o_proj") {
        layer->self_attn_o = ggml_dup_tensor(ctx, tensor);
        set_tensor(layer->self_attn_o, tensor);
    } else if (part == "cross_q_proj") {
        layer->cross_attn_q = ggml_dup_tensor(ctx, tensor);
        set_tensor(layer->cross_attn_q, tensor);
    } else if (part == "cross_k_proj") {
        layer->cross_attn_k = ggml_dup_tensor(ctx, tensor);
        set_tensor(layer->cross_attn_k, tensor);
    } else if (part == "cross_v_proj") {
        layer->cross_attn_v = ggml_dup_tensor(ctx, tensor);
        set_tensor(layer->cross_attn_v, tensor);
    } else if (part == "cross_o_proj") {
        layer->cross_attn_o = ggml_dup_tensor(ctx, tensor);
        set_tensor(layer->cross_attn_o, tensor);
    } else if (part == "pre_sa_norm") {
        layer->self_attn_norm = ggml_dup_tensor(ctx, tensor);
        set_tensor(layer->self_attn_norm, tensor);
    } else if (part == "pre_mlp_norm") {
        layer->mlp_norm = ggml_dup_tensor(ctx, tensor);
        set_tensor(layer->mlp_norm, tensor);
    } else if (part == "pre_ca_norm") {
        layer->cross_attn_norm = ggml_dup_tensor(ctx, tensor);
        set_tensor(layer->cross_attn_norm, tensor);
    } else if (part == "gate") {
        layer->gate = ggml_dup_tensor(ctx, tensor);
        set_tensor(layer->gate, tensor);
    } else if (part == "up") {
        layer->up = ggml_dup_tensor(ctx, tensor);
        set_tensor(layer->up, tensor);
    } else if (part == "wo") {
        layer->out = ggml_dup_tensor(ctx, tensor);
        set_tensor(layer->out, tensor);
    } else {
        TTS_ABORT("Unrecognized tensor '%s' for encoder layer when loading Dia from GGUF file.", part.c_str());
    }
}

void dia_model::assign_to_decoder(vector<string> parts, ggml_tensor * tensor, string name) {
    if (parts[2] == "embeddings") {
        TTS_ASSERT(parts.size() > 2);
        int index = stoi(parts[3]);
        TTS_ASSERT(index < decoder->embds.size());
        decoder->embds[index] = ggml_dup_tensor(ctx, tensor);
        set_tensor(decoder->embds[index], tensor);
    } else if (parts[2] == "norm") {
        decoder->norm = ggml_dup_tensor(ctx, tensor);
        set_tensor(decoder->norm, tensor);
    } else if (parts[2] == "heads") {
        TTS_ASSERT(parts.size() > 2);
        int index = stoi(parts[3]);
        TTS_ASSERT(index < decoder->heads.size());
        decoder->heads[index] = ggml_dup_tensor(ctx, tensor);
        set_tensor(decoder->heads[index], tensor);
    } else if (parts[2] == "layers") {
        TTS_ASSERT(parts.size() >= 4);
        int index = stoi(parts[3]);
        TTS_ASSERT(index < decoder->layers.size());
        assign_to_decoder_layer(parts[4], decoder->layers[index], tensor);
    } else {
        TTS_ABORT("Unrecognized tensor '%s' when loading Dia from GGUF file.", name.c_str());
    }
}
