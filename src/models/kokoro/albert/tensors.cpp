
void kokoro_model::assign_albert_weight(std::string name, ggml_tensor * tensor) {
    if (name == "embd") {
        embd_hidden = ggml_dup_tensor(ctx, tensor);
        set_tensor(embd_hidden, tensor);
    } else if (name == "embd_bias") {
        embd_hidden_bias = ggml_dup_tensor(ctx, tensor);
        set_tensor(embd_hidden_bias, tensor);
    } else if (name == "token_embd") {
        token_embd = ggml_dup_tensor(ctx, tensor);
        set_tensor(token_embd, tensor);
    } else if (name == "position_embd") {
        position_embd = ggml_dup_tensor(ctx, tensor);
        set_tensor(position_embd, tensor);
    } else if (name == "norm") {
        input_norm_weight = ggml_dup_tensor(ctx, tensor);
        set_tensor(input_norm_weight, tensor);
    } else if (name == "norm_bias") {
        input_norm_bias = ggml_dup_tensor(ctx, tensor);
        set_tensor(input_norm_bias, tensor);
    } else if (name == "token_type_embd") {
        static_token_type_values = ggml_dup_tensor(ctx, tensor);
        set_tensor(static_token_type_values, tensor);
    } else if (has_prefix(name, "layer")) {
        std::vector<std::string> parts = split(name, '.');
        int                      i     = std::stoi(parts[1]);
        if (parts[2] == "ffn") {
            layers[i]->ffn = ggml_dup_tensor(ctx, tensor);
            set_tensor(layers[i]->ffn, tensor);
        } else if (parts[2] == "ffn_bias") {
            layers[i]->ffn_bias = ggml_dup_tensor(ctx, tensor);
            set_tensor(layers[i]->ffn_bias, tensor);
        } else if (parts[2] == "ffn_out") {
            layers[i]->ffn_out = ggml_dup_tensor(ctx, tensor);
            set_tensor(layers[i]->ffn_out, tensor);
        } else if (parts[2] == "ffn_out_bias") {
            layers[i]->ffn_out_bias = ggml_dup_tensor(ctx, tensor);
            set_tensor(layers[i]->ffn_out_bias, tensor);
        } else if (parts[2] == "attn_norm") {
            layers[i]->layer_output_norm_weight = ggml_dup_tensor(ctx, tensor);
            set_tensor(layers[i]->layer_output_norm_weight, tensor);
        } else if (parts[2] == "attn_norm_bias") {
            layers[i]->layer_output_norm_bias = ggml_dup_tensor(ctx, tensor);
            set_tensor(layers[i]->layer_output_norm_bias, tensor);
        } else if (parts[2] == "q") {
            layers[i]->q = ggml_dup_tensor(ctx, tensor);
            set_tensor(layers[i]->q, tensor);
        } else if (parts[2] == "k") {
            layers[i]->k = ggml_dup_tensor(ctx, tensor);
            set_tensor(layers[i]->k, tensor);
        } else if (parts[2] == "v") {
            layers[i]->v = ggml_dup_tensor(ctx, tensor);
            set_tensor(layers[i]->v, tensor);
        } else if (parts[2] == "o") {
            layers[i]->o = ggml_dup_tensor(ctx, tensor);
            set_tensor(layers[i]->o, tensor);
        } else if (parts[2] == "q_bias") {
            layers[i]->q_bias = ggml_dup_tensor(ctx, tensor);
            set_tensor(layers[i]->q_bias, tensor);
        } else if (parts[2] == "k_bias") {
            layers[i]->k_bias = ggml_dup_tensor(ctx, tensor);
            set_tensor(layers[i]->k_bias, tensor);
        } else if (parts[2] == "v_bias") {
            layers[i]->v_bias = ggml_dup_tensor(ctx, tensor);
            set_tensor(layers[i]->v_bias, tensor);
        } else if (parts[2] == "o_bias") {
            layers[i]->o_bias = ggml_dup_tensor(ctx, tensor);
            set_tensor(layers[i]->o_bias, tensor);
        } else if (parts[2] == "ffn_norm") {
            layers[i]->attn_norm_weight = ggml_dup_tensor(ctx, tensor);
            set_tensor(layers[i]->attn_norm_weight, tensor);
        } else if (parts[2] == "ffn_norm_bias") {
            layers[i]->attn_norm_bias = ggml_dup_tensor(ctx, tensor);
            set_tensor(layers[i]->attn_norm_bias, tensor);
        }
    }
}
