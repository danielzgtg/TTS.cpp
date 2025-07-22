

enum t5_tensor {
    T5_EMBD,
    T5_NORM,
    T5_DOWN_PROJ,
    T5_DOWN_PROJ_BIAS,
    T5_RELATIVE_BIAS,
    T5_LAYER_ATTN_Q,
    T5_LAYER_ATTN_K,
    T5_LAYER_ATTN_V,
    T5_LAYER_ATTN_O,
    T5_LAYER_ATTN_NORM,
    T5_LAYER_WI_0,
    T5_LAYER_WI_1,
    T5_LAYER_WO,
    T5_LAYER_OUT_NORM,
};



static const std::map<std::string, t5_tensor> T5_TENSOR_GGUF_LOOKUP = {
    {"t5encoder.token_embd", T5_EMBD},
    {"t5encoder.enc.final_layer_norm", T5_NORM},
    {"t5encoder.down_proj", T5_DOWN_PROJ},
    {"t5encoder.down_proj_bias", T5_DOWN_PROJ_BIAS},
    {".attn_norm", T5_LAYER_ATTN_NORM},
    {".attn_q", T5_LAYER_ATTN_Q},
    {".attn_k", T5_LAYER_ATTN_K},
    {".attn_v", T5_LAYER_ATTN_V},
    {".attn_o", T5_LAYER_ATTN_O},
    {".attn_rel_b", T5_RELATIVE_BIAS},
    {".ffn_norm", T5_LAYER_OUT_NORM},
    {".ffn_gate", T5_LAYER_WI_1},
    {".ffn_down", T5_LAYER_WO},
    {".ffn_up", T5_LAYER_WI_0},
};

void assign_to_t5_layer(t5_encoder * model, t5_layer & layer, std::string name, ggml_tensor * tensor) {
    try {
        switch(T5_TENSOR_GGUF_LOOKUP.at(name)) {
            case T5_LAYER_ATTN_NORM:
                layer.attn_norm = ggml_dup_tensor(model->ctx, tensor);
                model->set_tensor(layer.attn_norm, tensor);
                break;
            case T5_LAYER_ATTN_Q:
                layer.q = ggml_dup_tensor(model->ctx, tensor);
                model->set_tensor(layer.q, tensor);
                break;
            case T5_LAYER_ATTN_K:
                layer.k = ggml_dup_tensor(model->ctx, tensor);
                model->set_tensor(layer.k, tensor);
                break;
            case T5_LAYER_ATTN_V:
                layer.v = ggml_dup_tensor(model->ctx, tensor);
                model->set_tensor(layer.v, tensor);
                break;
            case T5_LAYER_ATTN_O:
                layer.o = ggml_dup_tensor(model->ctx, tensor);
                model->set_tensor(layer.o, tensor);
                break;
            case T5_LAYER_OUT_NORM:
                layer.mlp_norm = ggml_dup_tensor(model->ctx, tensor);
                model->set_tensor(layer.mlp_norm, tensor);
                break;
            case T5_LAYER_WI_1:
                layer.wi_1 = ggml_dup_tensor(model->ctx, tensor);
                model->set_tensor(layer.wi_1, tensor);
                break;
            case T5_LAYER_WI_0:
                layer.wi_0 = ggml_dup_tensor(model->ctx, tensor);
                model->set_tensor(layer.wi_0, tensor);
                break;
            case T5_LAYER_WO:
                layer.wo = ggml_dup_tensor(model->ctx, tensor);
                model->set_tensor(layer.wo, tensor);
                break;
            case T5_RELATIVE_BIAS:
                model->relative_attn_bias = ggml_dup_tensor(model->ctx, tensor);
                model->set_tensor(model->relative_attn_bias, tensor);
                break;
            default:
                fprintf(stdout, "unassigned tensor %s\n", name.c_str());
                break;
        }
    } catch (const std::out_of_range& e) {
        TTS_ABORT("Error: %s\nTensor, '%s', is not a valid tensor.", e.what(), name.c_str());
    }
}

void assign_to_t5_encoder(t5_encoder * model, const std::string name, ggml_tensor * tensor) {
    if (tensor->data == NULL) {
        return;
    }
    std::string::size_type pos = name.find(".", 0);
    std::string top_level(name.substr(0, pos));
    if (T5_TENSOR_GGUF_LOOKUP.find(name) != T5_TENSOR_GGUF_LOOKUP.end()) {
        switch (T5_TENSOR_GGUF_LOOKUP.at(name)) {
            case T5_EMBD:
                model->embd = ggml_dup_tensor(model->ctx, tensor);
                model->set_tensor(model->embd, tensor);
                break;
            case T5_NORM:
                model->out_norm = ggml_dup_tensor(model->ctx, tensor);
                model->set_tensor(model->out_norm, tensor);
                break;
            case T5_DOWN_PROJ:
                model->down_proj = ggml_dup_tensor(model->ctx, tensor);
                model->set_tensor(model->down_proj, tensor);
                break;
            case T5_DOWN_PROJ_BIAS:
                model->down_proj_bias = ggml_dup_tensor(model->ctx, tensor);
                model->set_tensor(model->down_proj_bias, tensor);
                break;
            default:
                fprintf(stdout, "unassigned tensor %s\n", name.c_str());
                break;
        }
    } else if (top_level == "t5encoder") {
        auto pair = parse_layer_count(name, 2);
        int l = pair.first;
        std::string lt_name = pair.second;

        assign_to_t5_layer(model, model->layers[l], lt_name, tensor);
    } else {
        return;
    }
}


void t5_encoder::assign_weight(std::string name, ggml_tensor * tensor) {
    assign_to_t5_encoder(this, name, tensor);
}
