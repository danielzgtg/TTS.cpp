
void dia_model::assign_to_encoder_layer(string part, dia_encoder_layer * layer, ggml_tensor * tensor) {
    if (part == "q_proj") {
        layer->q = ggml_dup_tensor(ctx, tensor);
        set_tensor(layer->q, tensor);
    } else if (part == "k_proj") {
        layer->k = ggml_dup_tensor(ctx, tensor);
        set_tensor(layer->k, tensor);
    } else if (part == "v_proj") {
        layer->v = ggml_dup_tensor(ctx, tensor);
        set_tensor(layer->v, tensor);
    } else if (part == "o_proj") {
        layer->o = ggml_dup_tensor(ctx, tensor);
        set_tensor(layer->o, tensor);
    } else if (part == "pre_sa_norm") {
        layer->self_attn_norm = ggml_dup_tensor(ctx, tensor);
        set_tensor(layer->self_attn_norm, tensor);
    } else if (part == "post_sa_norm") {
        layer->mlp_norm = ggml_dup_tensor(ctx, tensor);
        set_tensor(layer->mlp_norm, tensor);
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

void dia_model::assign_to_encoder(vector<string> parts, ggml_tensor * tensor, string name) {
    if (parts[2] == "embedding") {
        encoder->embedding = ggml_dup_tensor(ctx, tensor);
        set_tensor(encoder->embedding, tensor);
    } else if (parts[2] == "norm") {
        encoder->norm = ggml_dup_tensor(ctx, tensor);
        set_tensor(encoder->norm, tensor);
    } else if (parts[2] == "layers") {
        TTS_ASSERT(parts.size() >= 4);
        int index = stoi(parts[3]);
        TTS_ASSERT(index < decoder->layers.size());
        assign_to_encoder_layer(parts[4], encoder->layers[index], tensor);
    } else {
        TTS_ABORT("Unrecognized tensor '%s' when loading Dia from GGUF file.", name.c_str());
    }
}
