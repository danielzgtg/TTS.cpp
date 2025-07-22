#include "model.h"
#include "util.h"

ggml_tensor * ada_residual_conv_block::build(ggml_context * ctx, ggml_tensor * x, ggml_tensor * style,
                                             ggml_tensor * sqrt_tensor) const {
    ggml_tensor * gamma = ggml_add(ctx, ggml_mul_mat(ctx, norm1_gamma, style), norm1_gamma_bias);
    ggml_tensor * beta  = ggml_add(ctx, ggml_mul_mat(ctx, norm1_beta, style), norm1_beta_bias);
    ggml_tensor * cur   = ggml_norm(ctx, x, 0.00001);

    // The addition between gamma * x and x is performed here because
    // ggml doesn't support scalar multiplication without initializing the scalars in advance.
    // An optimal remedy to this would be to increment the gamma bias above by one
    // when preparing the gguf file for the model.
    cur = ggml_add(ctx, cur, ggml_mul(ctx, cur, ggml_transpose(ctx, gamma)));
    cur = ggml_add(ctx, cur, ggml_transpose(ctx, beta));
    cur = ggml_leaky_relu(ctx, cur, 0.2f, false);

    if (pool) {
        cur = ggml_conv_transpose_1d(ctx, pool, cur, 2, 1, 1, 1, cur->ne[1]);
        cur = ggml_add(ctx, cur, pool_bias);
    }

    cur = ggml_conv_1d(ctx, conv1, cur, 1, 1, 1);

    cur   = ggml_add(ctx, cur, conv1_bias);
    gamma = ggml_add(ctx, ggml_mul_mat(ctx, norm2_gamma, style), norm2_gamma_bias);
    beta  = ggml_add(ctx, ggml_mul_mat(ctx, norm2_beta, style), norm2_beta_bias);
    cur   = ggml_norm(ctx, cur, 0.00001);

    // The addition between gamma * x and x is performed here
    // because ggml doesn't support scalar multiplication without initializing the scalars in advance.
    // An optimal remedy to this would be to increment the gamma bias above by one
    // when preparing the gguf file for the model.
    cur = ggml_add(ctx, cur, ggml_mul(ctx, cur, ggml_transpose(ctx, gamma)));
    cur = ggml_add(ctx, cur, ggml_transpose(ctx, beta));
    cur = ggml_leaky_relu(ctx, cur, 0.2f, false);
    cur = ggml_add(ctx, ggml_conv_1d(ctx, conv2, cur, 1, 1, 1), conv2_bias);

    ggml_tensor * res = cur;
    cur               = x;
    if (upsample) {
        cur = ggml_cont(ctx, ggml_transpose(ctx, cur));
        if (pool) {
            cur = ggml_upscale_ext(ctx, cur, cur->ne[0], cur->ne[1] * 2, cur->ne[2], cur->ne[3]);
        }
        cur = ggml_mul_mat(ctx, upsample, cur);
        cur = ggml_cont(ctx, ggml_transpose(ctx, cur));
    }
    cur = ggml_div(ctx, ggml_add(ctx, res, cur), sqrt_tensor);
    return cur;
}
