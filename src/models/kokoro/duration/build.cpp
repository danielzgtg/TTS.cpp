
#include "context.h"

kokoro_ubatch kokoro_duration_runner::build_worst_case_batch() {
    kokoro_ubatch batch;
    batch.n_tokens = model->max_context_length;
    return batch;
}

ggml_cgraph * kokoro_duration_runner::build_kokoro_duration_graph(kokoro_ubatch & batch) {
    init_build();
    // This '110000' number is coming from the number of nodes necessary
    // for the longest possible sequence computed by of the graph.
    // While it may be possible to precompute this by determining
    // the longest possible duration against the maximum context length of the model,
    // it is not easily performed given that nodes do not necessarily line up predictably
    // with the number of tensors in the model or its submodels.
    // In order to side step this problem I computed the graph and determined
    // the size in advance and use that constant value here.
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 110000, false);

    ggml_tensor * voice = model->voices[kctx->voice];
    ggml_tensor * cur;
    ggml_tensor * inpL;

    kctx->inp_tokens = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, batch.n_tokens);
    ggml_set_input(kctx->inp_tokens);

    if (!model->static_token_types) {
        kctx->token_types = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, batch.n_tokens);
        ggml_set_input(kctx->token_types);
    }

    kctx->positions = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, batch.n_tokens);
    ggml_set_input(kctx->positions);

    inpL = build_albert_inputs(ctx, &*model, kctx->inp_tokens, kctx->positions, kctx->token_types);
    ggml_set_name(inpL, "albert_embeddings");
    cur = inpL;

    ggml_tensor * KQ_mask_dec = build_albert_attn_mask(ctx, kctx, batch);

    for (int r = 0; r < model->n_recurrence; r++) {
        for (int l = 0; l < model->n_layers; l++) {
            ggml_tensor * residual = cur;
            ggml_tensor * attn_out;

            // self-attention
            {
                ggml_tensor * Qcur =
                    ggml_add(ctx, ggml_mul_mat(ctx, model->layers[l]->q, cur), model->layers[l]->q_bias);
                ggml_tensor * Kcur =
                    ggml_add(ctx, ggml_mul_mat(ctx, model->layers[l]->k, cur), model->layers[l]->k_bias);
                ggml_tensor * Vcur =
                    ggml_add(ctx, ggml_mul_mat(ctx, model->layers[l]->v, cur), model->layers[l]->v_bias);

                Qcur = ggml_reshape_3d(ctx, Qcur, model->head_size, model->n_attn_heads, batch.n_tokens);
                Kcur = ggml_reshape_3d(ctx, Kcur, model->head_size, model->n_attn_heads, batch.n_tokens);

                ggml_tensor * q  = ggml_permute(ctx, Qcur, 0, 2, 1, 3);
                ggml_tensor * k  = ggml_cont(ctx, ggml_permute(ctx, Kcur, 0, 2, 1, 3));
                ggml_tensor * kq = ggml_mul_mat(ctx, k, q);

                kq = ggml_soft_max_ext(ctx, kq, KQ_mask_dec, model->scale, 0.0f);

                ggml_tensor * v =
                    ggml_cont_3d(ctx, ggml_transpose(ctx, Vcur), batch.n_tokens, model->head_size, model->n_attn_heads);
                ggml_tensor * kqv        = ggml_mul_mat(ctx, kq, v);
                ggml_tensor * kqv_merged = ggml_permute(ctx, kqv, 2, 0, 1, 3);
                attn_out                 = ggml_cont_2d(ctx, kqv_merged, model->hidden_size, batch.n_tokens);
                attn_out = ggml_add(ctx, ggml_mul_mat(ctx, model->layers[l]->o, attn_out), model->layers[l]->o_bias);
            }
            cur = ggml_add(ctx, attn_out, residual);
            cur = build_albert_norm(ctx, cur, model->layers[l]->attn_norm_weight, model->layers[l]->attn_norm_bias);

            ggml_tensor * residualffn = cur;

            // ffn
            {
                cur = ggml_gelu(
                    ctx, ggml_add(ctx, ggml_mul_mat(ctx, model->layers[l]->ffn, cur), model->layers[l]->ffn_bias));
                cur = ggml_add(ctx, ggml_mul_mat(ctx, model->layers[l]->ffn_out, cur), model->layers[l]->ffn_out_bias);
            }

            cur = ggml_add(ctx, cur, residualffn);
            cur = build_albert_norm(ctx, cur, model->layers[l]->layer_output_norm_weight,
                                    model->layers[l]->layer_output_norm_bias);
        }
    }

    // duration / prosody prediction
    cur = ggml_add(ctx, ggml_mul_mat(ctx, model->prosody_pred->albert_encode, cur),
                   model->prosody_pred->albert_encode_bias);

    ggml_tensor * style_half =
        ggml_cont(ctx, ggml_view_1d(ctx, voice, voice->ne[0] / 2,
                                    voice->ne[0] / 2 * voice->nb[0] + (batch.n_tokens - 3) * voice->nb[1]));

    cur = ggml_concat(
        ctx, cur, ggml_repeat(ctx, style_half, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, style_half->ne[0], cur->ne[1])),
        0);

    for (auto l : model->prosody_pred->layers) {
        cur = build_lstm(ctx, cur, l->rnn, batch.n_tokens);

        ggml_tensor * gamma =
            ggml_add(ctx, ggml_mul_mat(ctx, l->ada_norm_gamma_weight, style_half), l->ada_norm_gamma_bias);
        ggml_tensor * beta =
            ggml_add(ctx, ggml_mul_mat(ctx, l->ada_norm_beta_weight, style_half), l->ada_norm_beta_bias);

        cur = ggml_norm(ctx, cur, 0.00001);

        // The addition between gamma * x and x is performed here because ggml doesn't support scalar multiplication without initializing the scalars in advance.
        // An optimal remedy to this would be to increment the gamma bias above by one when preparing the gguf file for the model.
        cur = ggml_add(ctx, ggml_add(ctx, cur, ggml_mul(ctx, cur, gamma)), beta);
        cur = ggml_concat(
            ctx, cur,
            ggml_repeat(ctx, style_half, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, style_half->ne[0], cur->ne[1])), 0);
        ggml_build_forward_expand(gf, cur);
    }

    ggml_tensor * d = ggml_cont(ctx, cur);
    ggml_set_name(d, "duration_hidden_states");
    ggml_build_forward_expand(gf, d);

    ggml_tensor * len;
    cur = build_lstm(ctx, cur, model->prosody_pred->duration_proj_lstm, batch.n_tokens);
    cur = ggml_sigmoid(ctx, ggml_add(ctx, ggml_mul_mat(ctx, model->prosody_pred->duration_proj, cur),
                                     model->prosody_pred->duration_proj_bias));
    // If we were to support speed we would add a constant tensor for the speed and divide here.
    len = ggml_round(ctx, ggml_sum_rows(ctx, cur));
    len = ggml_clamp(ctx, ggml_round(ctx, ggml_sum_rows(ctx, cur)), 1.0f, 50.0f);

    ggml_build_forward_expand(gf, len);

    free_build();

    return gf;
}
