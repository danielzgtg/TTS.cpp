#include "model.h"
#import "util.h"

static ggml_tensor * build_kokoro_generator_res_block(ggml_context * ctx, ggml_tensor * x, ggml_tensor * style,
                                                      kokoro_generator_residual_block * block) {
    ggml_tensor * cur;
    ggml_tensor * gamma;
    ggml_tensor * beta;
    ggml_tensor * inpl = x;
    for (int i = 0; i < block->convs1_weights.size(); i++) {
        gamma = ggml_add(ctx, ggml_mul_mat(ctx, block->adain1d_1_gamma_weights[i], style),
                         block->adain1d_1_gamma_biases[i]);
        beta =
            ggml_add(ctx, ggml_mul_mat(ctx, block->adain1d_1_beta_weights[i], style), block->adain1d_1_beta_biases[i]);
        cur = ggml_cont(ctx, ggml_transpose(ctx, ggml_norm(ctx, inpl, 0.00001)));

        // The addition between gamma * x and x is performed here because ggml doesn't support scalar multiplication without initializing the scalars in advance.
        // An optimal remedy to this would be to increment the gamma bias above by one when preparing the gguf file for the model.
        cur = ggml_add(ctx, ggml_add(ctx, cur, ggml_mul(ctx, cur, gamma)), beta);
        cur = snake_1d(ctx, block->input_alphas[i], ggml_cont(ctx, ggml_transpose(ctx, cur)));

        cur = ggml_add(
            ctx,
            ggml_conv_1d(ctx, block->convs1_weights[i], cur, 1, block->conv1_paddings[i], block->conv1_dilations[i]),
            block->convs1_biases[i]);
        gamma = ggml_add(ctx, ggml_mul_mat(ctx, block->adain1d_2_gamma_weights[i], style),
                         block->adain1d_2_gamma_biases[i]);
        beta =
            ggml_add(ctx, ggml_mul_mat(ctx, block->adain1d_2_beta_weights[i], style), block->adain1d_2_beta_biases[i]);
        cur = ggml_cont(ctx, ggml_transpose(ctx, ggml_norm(ctx, cur, 0.00001)));

        // The addition between gamma * x and x is performed here because ggml doesn't support scalar multiplication without initializing the scalars in advance.
        // An optimal remedy to this would be to increment the gamma bias above by one when preparing the gguf file for the model.
        cur = ggml_cont(ctx, ggml_transpose(ctx, ggml_add(ctx, ggml_add(ctx, cur, ggml_mul(ctx, cur, gamma)), beta)));

        cur  = snake_1d(ctx, block->output_alphas[i], cur);
        cur  = ggml_add(ctx, ggml_conv_1d(ctx, block->convs2_weights[i], cur, 1, block->conv1_paddings[0], 1),
                        block->convs2_biases[i]);
        inpl = ggml_add(ctx, inpl, cur);
    }
    return inpl;
}

static ggml_tensor * build_noise_block(ggml_context * ctx, kokoro_noise_residual_block * block, ggml_tensor * x,
                                       ggml_tensor * style) {
    // This conv_1d seems replaceable with squeezed and transposed ggml_mul_mut, but s0 and p0 are dynamic
    ggml_tensor * cur =
        ggml_add(ctx, ggml_conv_1d(ctx, block->input_conv, x, block->input_conv_stride, block->input_conv_padding, 1),
                 block->input_conv_bias);
    return build_kokoro_generator_res_block(ctx, cur, style, block->res_block);
}

static ggml_tensor * build_sin_gen(ggml_context * ctx, kokoro_model * model, kokoro_context * kctx, ggml_tensor * x,
                                   int harmonic_num, int sequence_length, float voice_threshold, float sin_amp,
                                   float noise_std) {
    ggml_tensor * cur =
        ggml_mul(ctx, ggml_repeat(ctx, x, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, x->ne[0], harmonic_num)),
                 model->harmonic_sampling_norm);
    cur                    = ggml_mul(ctx, ggml_cumsum(ctx, ggml_mod(ctx, cur, 1.0f)), model->sampling_factor_scalar);
    cur                    = ggml_upscale_linear(ctx, cur, 300);
    ggml_tensor * upscaled = ggml_upscale_ext(ctx, x, x->ne[0] * 300, x->ne[1], x->ne[2], x->ne[3]);

    kctx->uv_noise_data = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, sequence_length * harmonic_num + 4);
    ggml_set_input(kctx->uv_noise_data);

    ggml_tensor * fake = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, sequence_length, harmonic_num, 2);

    // ggml doesn't support boolean tensors nor does it support greater than and roll ops. As a result, we represent these boolean tensors as 1.0 or 0.0 or simply perform
    // multiplications in place via a custom map.
    ggml_tensor * uv_noise =
        ggml_map_custom3(ctx, fake, upscaled, kctx->uv_noise_data, &uv_noise_compute, sequence_length, nullptr);

    ggml_tensor * noise =
        ggml_cont(ctx, ggml_view_2d(ctx, uv_noise, uv_noise->ne[0], uv_noise->ne[1], uv_noise->nb[1], uv_noise->nb[2]));
    ggml_tensor * uv =
        ggml_cont(ctx, ggml_view_2d(ctx, uv_noise, uv_noise->ne[0], uv_noise->ne[1], uv_noise->nb[1], 0));

    return ggml_cont(ctx, ggml_transpose(ctx, ggml_add(ctx, ggml_mul(ctx, ggml_sin(ctx, cur), uv), noise)));
}

ggml_tensor * build_generator(ggml_context * ctx, kokoro_model * model, kokoro_context * kctx, ggml_tensor * x,
                              ggml_tensor * style, ggml_tensor * f0_curve, kokoro_generator * generator,
                              int sequence_length, ggml_tensor * window_sq_sum, ggml_cgraph * gf) {
    ggml_tensor * sing = build_sin_gen(ctx, model, kctx, f0_curve, model->harmonic_num + 1, f0_curve->ne[0] * 300,
                                       model->voice_threshold, model->sin_amp, model->noise_std);
    ggml_tensor * har =
        ggml_tanh(ctx, ggml_add(ctx, ggml_mul_mat(ctx, generator->m_source_weight, sing), generator->m_source_bias));

    har = stft(ctx, ggml_cont(ctx, ggml_transpose(ctx, har)), generator->window, model->true_n_fft, model->stft_hop,
               true, true);

    // STFT returns a vector of shape [nFFT, frames, batch, 2] where the final shape (2) separates the magnitude and the phase
    // kokoro concatenates the n_fft from the magnitude and the phase together so we have to split them up and concatenate
    // along the n_fft axis
    ggml_tensor * mhar =
        ggml_cont(ctx, ggml_view_3d(ctx, har, har->ne[0], har->ne[1], har->ne[2], har->nb[1], har->nb[2], 0));
    ggml_tensor * phhar =
        ggml_cont(ctx, ggml_view_3d(ctx, har, har->ne[0], har->ne[1], har->ne[2], har->nb[1], har->nb[2], har->nb[3]));
    ggml_tensor * combined_har = ggml_cont(ctx, ggml_transpose(ctx, ggml_concat(ctx, mhar, phhar, 0)));

    ggml_tensor * cur = x;
    for (int i = 0; i < generator->ups.size(); i++) {
        cur = ggml_leaky_relu(ctx, cur, 0.1f, false);
        cur = ggml_add(
            ctx,
            ggml_conv_transpose_1d(ctx, generator->ups[i]->upsample_weight, ggml_cont(ctx, ggml_transpose(ctx, cur)),
                                   generator->ups[i]->stride, generator->ups[i]->padding, 1, 0, 1),
            generator->ups[i]->upsample_bias);
        if (i == generator->ups.size() - 1) {
            // This is a hacky way of implementing the simple reflection padding used here.
            // In general, ggml should eventually be built to support expressive reflective padding but for such simple front padding this makes more sense.
            ggml_tensor * temp =
                ggml_cont(ctx, ggml_view_3d(ctx, cur, 1, cur->ne[1], cur->ne[2], cur->nb[1], cur->nb[2], cur->nb[0]));
            cur = ggml_concat(ctx, temp, cur, 0);
        }
        ggml_tensor * x_source =
            build_noise_block(ctx, generator->noise_blocks[i], ggml_cont(ctx, combined_har), style);
        cur             = ggml_add(ctx, cur, x_source);
        ggml_tensor * x = cur;
        for (int ii = 0; ii < model->n_kernels; ii++) {
            if (ii == 0) {
                cur = build_kokoro_generator_res_block(ctx, x, style, generator->res_blocks[i * model->n_kernels + ii]);
            } else {
                cur = ggml_add(
                    ctx, cur,
                    build_kokoro_generator_res_block(ctx, x, style, generator->res_blocks[i * model->n_kernels + ii]));
            }
        }
        cur = ggml_cont(ctx, ggml_transpose(ctx, ggml_div(ctx, cur, model->n_kernels_tensor)));
    }

    cur = ggml_leaky_relu(ctx, cur, 0.01f, false);
    cur = ggml_add(ctx,
                   ggml_conv_1d(ctx, generator->out_conv_weight, ggml_cont(ctx, ggml_transpose(ctx, cur)), 1,
                                model->out_conv_padding, 1),
                   generator->out_conv_bias);

    ggml_tensor * spec  = ggml_view_3d(ctx, cur, cur->ne[0], model->post_n_fft, cur->ne[2], cur->nb[1], cur->nb[2], 0);
    ggml_tensor * phase = ggml_view_3d(ctx, cur, cur->ne[0], cur->ne[1] - model->post_n_fft, cur->ne[2], cur->nb[1],
                                       cur->nb[2], cur->nb[1] * model->post_n_fft);
    phase               = ggml_sin(ctx, phase);
    spec                = ggml_exp(ctx, spec);

    cur = ggml_concat(ctx, spec, phase, 3);  // istft expects the magnitude and phase concatenated after the batch;
    cur = istft(ctx, ggml_cont(ctx, ggml_transpose(ctx, cur)), window_sq_sum, generator->window, model->true_n_fft,
                model->stft_hop, true, true);
    ggml_set_name(cur, "after_res_gen");
    ggml_build_forward_expand(gf, cur);
    return cur;
}
