#include "model.h"
#include "util.h"

void duration_predictor::assign_duration_weight(ggml_context * ctx, string name, ggml_tensor * tensor) {
    if (name == "encode") {
        albert_encode = ggml_dup_tensor(ctx, tensor);
        set_tensor(albert_encode, tensor);
    } else if (name == "encode_bias") {
        albert_encode_bias = ggml_dup_tensor(ctx, tensor);
        set_tensor(albert_encode_bias, tensor);
    } else if (name == "duration_proj") {
        duration_proj = ggml_dup_tensor(ctx, tensor);
        set_tensor(duration_proj, tensor);
    } else if (name == "duration_proj_bias") {
        duration_proj_bias = ggml_dup_tensor(ctx, tensor);
        set_tensor(duration_proj_bias, tensor);
    } else if (name == "n_proj_kernel") {
        tensor        = squeeze_3d_2d_e0(ctx, tensor);
        n_proj_kernel = ggml_dup_tensor(ctx, tensor);
        set_tensor(n_proj_kernel, tensor);
    } else if (name == "n_proj_bias") {
        n_proj_bias = ggml_dup_tensor(ctx, ggml_transpose(ctx, tensor));
        set_tensor(n_proj_bias, tensor);
    } else if (name == "f0_proj_kernel") {
        tensor         = squeeze_3d_2d_e0(ctx, tensor);
        f0_proj_kernel = ggml_dup_tensor(ctx, tensor);
        set_tensor(f0_proj_kernel, tensor);
    } else if (name == "f0_proj_bias") {
        f0_proj_bias = ggml_dup_tensor(ctx, ggml_transpose(ctx, tensor));
        set_tensor(f0_proj_bias, tensor);
    } else {
        std::vector<std::string> parts = split(name, ".");
        if (parts[0] == "shared_lstm") {
            assign_lstm(shared_lstm, name.substr(parts[0].size() + 1), tensor);
        } else if (parts[0] == "duration_lstm") {
            assign_lstm(duration_proj_lstm, name.substr(parts[0].size() + 1), tensor);
        } else if (parts[0] == "f0_blocks") {
            int i = std::stoi(parts[1]);
            assign_ada_res_block(f0_blocks[i], parts[2], tensor);
        } else if (parts[0] == "n_blocks") {
            int i = std::stoi(parts[1]);
            assign_ada_res_block(n_blocks[i], parts[2], tensor);
        } else if (parts[0] == "layers") {
            int i = std::stoi(parts[1]);
            i     = i / 2;
            if (parts[2] == "gamma_weight") {
                layers[i]->ada_norm_gamma_weight = ggml_dup_tensor(ctx, tensor);
                set_tensor(layers[i]->ada_norm_gamma_weight, tensor);
            } else if (parts[2] == "gamma_bias") {
                layers[i]->ada_norm_gamma_bias = ggml_dup_tensor(ctx, tensor);
                set_tensor(layers[i]->ada_norm_gamma_bias, tensor);
            } else if (parts[2] == "beta_weight") {
                layers[i]->ada_norm_beta_weight = ggml_dup_tensor(ctx, tensor);
                set_tensor(layers[i]->ada_norm_beta_weight, tensor);
            } else if (parts[2] == "beta_bias") {
                layers[i]->ada_norm_beta_bias = ggml_dup_tensor(ctx, tensor);
                set_tensor(layers[i]->ada_norm_beta_bias, tensor);
            } else if (parts[2] == "lstm") {
                assign_lstm(layers[i]->rnn, name.substr(parts[0].size() + parts[1].size() + parts[2].size() + 3),
                            tensor);
            }
        }
    }
}
