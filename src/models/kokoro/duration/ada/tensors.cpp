#include "buffers.h"
#include "model.h"
#include "util.h"

void ada_residual_conv_block::assign_ada_res_block(tts_buffer & ctx, string_view name, ggml_tensor * tensor) {
    if (name == "norm1_gamma_weight") {
        norm1_gamma = ggml_dup_tensor(ctx, tensor);
        ctx.set_tensor(norm1_gamma, tensor);
    } else if (name == "norm2_gamma_weight") {
        norm2_gamma = ggml_dup_tensor(ctx, tensor);
        ctx.set_tensor(norm2_gamma, tensor);
    } else if (name == "norm1_gamma_bias") {
        norm1_gamma_bias = ggml_dup_tensor(ctx, tensor);
        ctx.set_tensor(norm1_gamma_bias, tensor);
    } else if (name == "norm2_gamma_bias") {
        norm2_gamma_bias = ggml_dup_tensor(ctx, tensor);
        ctx.set_tensor(norm2_gamma_bias, tensor);
    } else if (name == "norm1_beta_weight") {
        norm1_beta = ggml_dup_tensor(ctx, tensor);
        ctx.set_tensor(norm1_beta, tensor);
    } else if (name == "norm2_beta_weight") {
        norm2_beta = ggml_dup_tensor(ctx, tensor);
        ctx.set_tensor(norm2_beta, tensor);
    } else if (name == "norm1_beta_bias") {
        norm1_beta_bias = ggml_dup_tensor(ctx, tensor);
        ctx.set_tensor(norm1_beta_bias, tensor);
    } else if (name == "norm2_beta_bias") {
        norm2_beta_bias = ggml_dup_tensor(ctx, tensor);
        ctx.set_tensor(norm2_beta_bias, tensor);
    } else if (name == "conv1_weight") {
        conv1 = ggml_dup_tensor(ctx, tensor);
        ctx.set_tensor(conv1, tensor);
    } else if (name == "conv2_weight") {
        conv2 = ggml_dup_tensor(ctx, tensor);
        ctx.set_tensor(conv2, tensor);
    } else if (name == "conv1_bias") {
        conv1_bias = ggml_dup_tensor(ctx, ggml_transpose(ctx, tensor));
        ctx.set_tensor(conv1_bias, tensor);
    } else if (name == "conv2_bias") {
        conv2_bias = ggml_dup_tensor(ctx, ggml_transpose(ctx, tensor));
        ctx.set_tensor(conv2_bias, tensor);
    } else if (name == "pool_weight") {
        pool = ggml_dup_tensor(ctx, tensor);
        ctx.set_tensor(pool, tensor);
    } else if (name == "pool_bias") {
        pool_bias = ggml_dup_tensor(ctx, ggml_transpose(ctx, tensor));
        ctx.set_tensor(pool_bias, tensor);
    } else if (name == "conv1x1_weight") {
        tensor   = squeeze_3d_2d_e0(ctx, tensor);
        upsample = ggml_dup_tensor(ctx, tensor);
        ctx.set_tensor(upsample, tensor);
    } else if (name == "conv1x1_bias") {
        upsample_bias = ggml_dup_tensor(ctx, ggml_transpose(ctx, tensor));
        ctx.set_tensor(upsample_bias, tensor);
    }
}
