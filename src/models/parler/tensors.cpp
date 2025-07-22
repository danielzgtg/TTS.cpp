

enum parler_tensor {
    PARLER_EMBD,
    PARLER_EMBD_PROMPTS,
    PARLER_TEXT_ENCODING,
    PARLER_POSITIONAL_EMBD,
    PARLER_HEAD,
    PARLER_NORM,
    PARLER_NORM_BIAS,
    PARLER_LAYER_SELF_ATTN_Q,
    PARLER_LAYER_SELF_ATTN_K,
    PARLER_LAYER_SELF_ATTN_V,
    PARLER_LAYER_SELF_ATTN_O,
    PARLER_LAYER_SELF_ATTN_NORM,
    PARLER_LAYER_SELF_ATTN_NORM_BIAS,
    PARLER_LAYER_ATTN_Q,
    PARLER_LAYER_ATTN_K,
    PARLER_LAYER_ATTN_V,
    PARLER_LAYER_ATTN_O,
    PARLER_LAYER_ATTN_NORM,
    PARLER_LAYER_ATTN_NORM_BIAS,
    PARLER_LAYER_FC1,
    PARLER_LAYER_FC2,
    PARLER_LAYER_OUT_NORM,
    PARLER_LAYER_OUT_NORM_BIAS,
};

// For loading parler model from gguf file.
static const std::map<std::string, parler_tensor> PARLER_TENSOR_GGUF_LOOKUP = {
    {"layer_norm.weight", PARLER_NORM},
    {"layer_norm.bias", PARLER_NORM_BIAS},
    {"embed_prompts", PARLER_EMBD_PROMPTS},
    {"text_encoding", PARLER_TEXT_ENCODING},
    {"positional_embed", PARLER_POSITIONAL_EMBD},
    {".self_attn.q_proj.weight", PARLER_LAYER_SELF_ATTN_Q},
    {".self_attn.k_proj.weight", PARLER_LAYER_SELF_ATTN_K},
    {".self_attn.v_proj.weight", PARLER_LAYER_SELF_ATTN_V},
    {".self_attn.out_proj.weight", PARLER_LAYER_SELF_ATTN_O},
    {".self_attn_layer_norm.weight", PARLER_LAYER_SELF_ATTN_NORM},
    {".self_attn_layer_norm.bias", PARLER_LAYER_SELF_ATTN_NORM_BIAS},
    {".encoder_attn.q_proj.weight", PARLER_LAYER_ATTN_Q},
    {".encoder_attn.k_proj.weight", PARLER_LAYER_ATTN_K},
    {".encoder_attn.v_proj.weight", PARLER_LAYER_ATTN_V},
    {".encoder_attn.out_proj.weight", PARLER_LAYER_ATTN_O},
    {".encoder_attn_layer_norm.weight", PARLER_LAYER_ATTN_NORM},
    {".encoder_attn_layer_norm.bias", PARLER_LAYER_ATTN_NORM_BIAS},
    {".fc1.weight", PARLER_LAYER_FC1},
    {".fc2.weight", PARLER_LAYER_FC2},
    {".final_layer_norm.weight", PARLER_LAYER_OUT_NORM},
    {".final_layer_norm.bias", PARLER_LAYER_OUT_NORM_BIAS},
    {".weight", PARLER_EMBD},
    {".weight.head", PARLER_HEAD}
};

void parler_tts_model::assign_weight(std::string name, ggml_tensor * tensor) {
    assign_to_decoder(this, name, tensor);
}


void assign_parler_layer(parler_tts_model * model, parler_layer * layer, std::string name, ggml_tensor * tensor) {
    try {
        switch(PARLER_TENSOR_GGUF_LOOKUP.at(name)) {
            case PARLER_LAYER_SELF_ATTN_Q:
                layer->self_attn_q_proj = ggml_dup_tensor(model->ctx, tensor);
                model->set_tensor(layer->self_attn_q_proj, tensor);
                break;
            case PARLER_LAYER_SELF_ATTN_K:
                layer->self_attn_k_proj = ggml_dup_tensor(model->ctx, tensor);
                model->set_tensor(layer->self_attn_k_proj, tensor);
                break;
            case PARLER_LAYER_SELF_ATTN_V:
                layer->self_attn_v_proj = ggml_dup_tensor(model->ctx, tensor);
                model->set_tensor(layer->self_attn_v_proj, tensor);
                break;
            case PARLER_LAYER_SELF_ATTN_O:
                layer->self_attn_o_proj = ggml_dup_tensor(model->ctx, tensor);
                model->set_tensor(layer->self_attn_o_proj, tensor);
                break;
            case PARLER_LAYER_SELF_ATTN_NORM:
                layer->self_attn_norm = ggml_dup_tensor(model->ctx, tensor);
                model->set_tensor(layer->self_attn_norm, tensor);
                break;
            case PARLER_LAYER_SELF_ATTN_NORM_BIAS:
                layer->self_attn_norm_bias = ggml_dup_tensor(model->ctx, tensor);
                model->set_tensor(layer->self_attn_norm_bias, tensor);
                break;
            case PARLER_LAYER_ATTN_Q:
                if (model->use_cross_attn) {
                    layer->attn_q_proj = ggml_dup_tensor(model->ctx, tensor);
                    model->set_tensor(layer->attn_q_proj, tensor);
                }
                break;
            case PARLER_LAYER_ATTN_K:
                if (model->use_cross_attn) {
                    layer->attn_k_proj = ggml_dup_tensor(model->ctx, tensor);
                    model->set_tensor(layer->attn_k_proj, tensor);
                }
                break;
            case PARLER_LAYER_ATTN_V:
                if (model->use_cross_attn) {
                    layer->attn_v_proj = ggml_dup_tensor(model->ctx, tensor);
                    model->set_tensor(layer->attn_v_proj, tensor);
                }
                break;
            case PARLER_LAYER_ATTN_O:
                if (model->use_cross_attn) {
                    layer->attn_o_proj = ggml_dup_tensor(model->ctx, tensor);
                    model->set_tensor(layer->attn_o_proj, tensor);
                }
                break;
            case PARLER_LAYER_ATTN_NORM:
                if (model->use_cross_attn) {
                    layer->attn_norm = ggml_dup_tensor(model->ctx, tensor);
                    model->set_tensor(layer->attn_norm, tensor);
                }
                break;
            case PARLER_LAYER_ATTN_NORM_BIAS:
                if (model->use_cross_attn) {
                    layer->attn_norm_bias = ggml_dup_tensor(model->ctx, tensor);
                    model->set_tensor(layer->attn_norm_bias, tensor);
                }
                break;
            case PARLER_LAYER_FC1:
                layer->fc1 = ggml_dup_tensor(model->ctx, tensor);
                model->set_tensor(layer->fc1, tensor);
                break;
            case PARLER_LAYER_FC2:
                layer->fc2 = ggml_dup_tensor(model->ctx, tensor);
                model->set_tensor(layer->fc2, tensor);
                break;
            case PARLER_LAYER_OUT_NORM:
                layer->final_norm = ggml_dup_tensor(model->ctx, tensor);
                model->set_tensor(layer->final_norm, tensor);
                break;
            case PARLER_LAYER_OUT_NORM_BIAS:
                layer->final_norm_bias = ggml_dup_tensor(model->ctx, tensor);
                model->set_tensor(layer->final_norm_bias, tensor);
                break;
            default:
                fprintf(stdout, "unassigned tensor %s\n", name.c_str());
                break;
        }
    } catch (const std::out_of_range& e) {
        TTS_ABORT("Error: %s\nTensor, '%s', is not a valid tensor.", e.what(), name.c_str());
    }
}

void assign_to_decoder(parler_tts_model * model, const std::string name, ggml_tensor * tensor) {
    if (PARLER_TENSOR_GGUF_LOOKUP.find(name) != PARLER_TENSOR_GGUF_LOOKUP.end()) {
        try {
            switch (PARLER_TENSOR_GGUF_LOOKUP.at(name)) {
                case PARLER_NORM:
                    model->layer_norm = ggml_dup_tensor(model->ctx, tensor);
                    model->set_tensor(model->layer_norm, tensor);
                    break;
                case PARLER_NORM_BIAS:
                    model->layer_norm_bias = ggml_dup_tensor(model->ctx, tensor);
                    model->set_tensor(model->layer_norm_bias, tensor);
                    break;
                case PARLER_EMBD_PROMPTS:
                    model->prompt_embd = ggml_dup_tensor(model->ctx, tensor);
                    model->set_tensor(model->prompt_embd, tensor);
                    break;
                case PARLER_TEXT_ENCODING:
                    if (model->use_cross_attn) {
                        model->precomputed_input_emb = ggml_dup_tensor(model->ctx, tensor);
                        model->set_tensor(model->precomputed_input_emb, tensor);
                    }
                    break;
                case PARLER_POSITIONAL_EMBD:
                    model->precomputed_positional_embds = ggml_dup_tensor(model->ctx, tensor);
                    model->set_tensor(model->precomputed_positional_embds, tensor);
                    break;
                default:
                    fprintf(stdout, "unassigned tensor %s\n", name.c_str());
                    break;
            }
        } catch (const std::out_of_range& e) {
            TTS_ABORT("Error: %s\nTensor, '%s', is not a valid tensor.", e.what(), name.c_str());
        }
    } else if (std::find_if(name.begin(), name.end(), ::isdigit) != name.end())  {
        auto pair = parse_layer_count(name);
        int layer = pair.first;
        std::string lt_name = pair.second;
        if (name.find("embed_tokens") != std::string::npos) {
            model->embds[layer] = ggml_dup_tensor(model->ctx, tensor);
            model->set_tensor(model->embds[layer], tensor);
        } else if (name.find("lm_heads") != std::string::npos) {
            model->heads[layer] = ggml_dup_tensor(model->ctx, tensor);
            model->set_tensor(model->heads[layer], tensor);
        } else {
            assign_parler_layer(model, model->layers[layer], lt_name, tensor);
        }
    }
}


void parler_tts_runner::assign_weight(std::string name, ggml_tensor * tensor) {
    std::string::size_type pos = name.find(".", 0);
    std::string top_level(name.substr(0, pos));
    std::string value(name.substr(pos + 1));
    if (tensor->data == NULL) {
        return;
    }
    if (top_level == "audio_encoder") {
        dac_runner->model->assign_weight(value, tensor);
    } else if (top_level == "decoder") {
        model->assign_weight(value, tensor);
    } else {
        return;
    }
}
