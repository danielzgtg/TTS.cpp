#include "ggml.h"

static ggml_tensor * build_dia_decoder_inp_embd(ggml_context * ctx, dia_context * dctx, dia_decoder * decoder,
                                                dia_ubatch & batch, uint32_t n_output_heads) {
    ggml_tensor * input_embs;

    dctx->audio_inp_tokens = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_output_heads * 2);
    ggml_set_input(dctx->audio_inp_tokens);
    for (int i = 0; i < n_output_heads; i++) {
        ggml_tensor * view =
            ggml_view_1d(ctx, dctx->audio_inp_tokens, 2, i * ggml_element_size(dctx->audio_inp_tokens));
        view->nb[0] = n_output_heads * ggml_element_size(dctx->audio_inp_tokens);
        if (i == 0) {
            input_embs = ggml_get_rows(ctx, decoder->embds[i], view);
        } else {
            input_embs = ggml_add(ctx, ggml_get_rows(ctx, decoder->embds[i], view), input_embs);
        }
    }
    return input_embs;
}

static ggml_tensor * build_dia_head_outputs(ggml_context * ctx, dia_model * model, ggml_tensor * cur) {
    // going to cat the heads together and then reshape them
    ggml_tensor * out;
    for (int i = 0; i < model->n_output_heads; i++) {
        if (i == 0) {
            out = ggml_mul_mat(ctx, model->decoder->heads[i], cur);
        } else {
            out = ggml_concat(ctx, out, ggml_mul_mat(ctx, model->decoder->heads[i], cur), 2);
        }
    }
    ggml_tensor * cond   = ggml_cont(ctx, ggml_view_2d(ctx, out, out->ne[0], out->ne[2], out->nb[2], 0));
    ggml_tensor * uncond = ggml_cont(ctx, ggml_view_2d(ctx, out, out->ne[0], out->ne[2], out->nb[2], out->nb[1]));
    return ggml_map_custom2(ctx, cond, uncond, &cfg_scale, out->ne[0], &model->cfg_scale_data);
}

static ggml_tensor * build_dia_decoder(ggml_cgraph * gf, ggml_context * ctx, dia_model * model, dia_context * dctx,
                                       dia_kv_cache * cache, dia_ubatch & batch, ggml_tensor * encoder_hidden_states) {
    dctx->positions = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, batch.sequence_length);
    ggml_set_input(dctx->positions);
    ggml_tensor * cur = build_dia_decoder_inp_embd(ctx, dctx, model->decoder, batch, model->n_output_heads);

    for (int l = 0; l < model->decoder->layers.size(); l++) {
        dia_decoder_layer * layer    = model->decoder->layers[l];
        ggml_tensor *       residual = cur;

        cur = dia_layer_norm(ctx, cur, layer->self_attn_norm);
        // self-attention
        {
            ggml_tensor * Qcur = ggml_mul_mat(ctx, layer->self_attn_q, cur);
            ggml_tensor * Kcur = ggml_mul_mat(ctx, layer->self_attn_k, cur);
            ggml_tensor * Vcur = ggml_mul_mat(ctx, layer->self_attn_v, cur);

            build_dia_self_kv_store(ctx, dctx, model, cache, gf, Kcur, Vcur, batch, l);
            ggml_tensor * k =
                ggml_view_4d(ctx, cache->k_l[l], model->head_size, model->decoder_attn_heads,
                             dctx->current_position + 1, 2, ggml_element_size(cache->k_l[l]) * model->head_size,
                             ggml_element_size(cache->k_l[l]) * model->decoder_attn_heads * model->head_size,
                             ggml_element_size(cache->k_l[l]) * model->decoder_attn_heads * model->head_size *
                                 model->max_generation_size,
                             0);
            k = ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3));

            ggml_tensor * v = ggml_view_3d(
                ctx, cache->v_l[l], model->head_size * model->decoder_attn_heads, dctx->current_position + 1, 2,
                ggml_element_size(cache->v_l[l]) * model->decoder_attn_heads * model->head_size,
                ggml_element_size(cache->v_l[l]) * model->decoder_attn_heads * model->head_size *
                    model->max_generation_size,
                0);
            v = ggml_cont_4d(ctx, ggml_transpose(ctx, v), dctx->current_position + 1, model->head_size,
                             model->decoder_attn_heads, 2);

            // As noted in the encoder Dia uses the Neo-X protocol for RoPE.
            Qcur             = ggml_rope(ctx,
                                         ggml_cont(ctx, ggml_reshape_4d(ctx, Qcur, model->head_size, model->decoder_attn_heads,
                                                                        batch.sequence_length, 2)),
                                         dctx->positions, model->head_size, 2);
            ggml_tensor * q  = ggml_cont(ctx, ggml_permute(ctx, Qcur, 0, 2, 1, 3));
            ggml_tensor * kq = ggml_mul_mat(ctx, ggml_cont(ctx, k), q);

            // given that attention bias, scaling and masking are not used for decoding, it might be faster to prefer the #ggml_soft_max op here,
            kq                       = ggml_soft_max_ext(ctx, kq, nullptr, 1.0f, 0.0f);
            ggml_tensor * kqv        = ggml_mul_mat(ctx, kq, v);
            ggml_tensor * kqv_merged = ggml_cont(ctx, ggml_permute(ctx, kqv, 2, 0, 1, 3));
            cur = ggml_cont_3d(ctx, kqv_merged, model->decoder_hidden_size, batch.sequence_length, 2);
            cur = ggml_mul_mat(ctx, layer->self_attn_o, cur);
        }

        // if we ever need to support multiple step decoder runs then this reshape will need to be replaced with permutation.
        cur                          = ggml_cont_2d(ctx, cur, cur->ne[0], 2);
        cur                          = ggml_add(ctx, cur, residual);
        ggml_tensor * residual_cross = cur;

        cur = dia_layer_norm(ctx, cur, layer->cross_attn_norm);
        // cross-attention
        {
            ggml_tensor * cross_Qcur = ggml_mul_mat(ctx, layer->cross_attn_q, cur);

            // only load the cross attention kv store when performing the encoding step
            if (batch.encoder_step) {
                build_dia_cross_kv_store(ctx, dctx, model, cache, gf, encoder_hidden_states, l);
            }

            ggml_tensor * cross_k = ggml_view_4d(
                ctx, cache->cross_k_l[l], model->head_size, model->decoder_attn_heads, 2,
                model->max_encoder_context_length, model->head_size * ggml_element_size(cache->cross_k_l[l]),
                model->head_size * model->decoder_attn_heads * ggml_element_size(cache->cross_k_l[l]),
                model->head_size * model->decoder_attn_heads * 2 * ggml_element_size(cache->cross_k_l[l]), 0);
            // the double permute operation shouldn't be necessary here, but it seems that currently ggml permute only currently alows for a single
            // axis pair to be transposed.
            cross_k = ggml_cont(ctx, ggml_permute(ctx, ggml_permute(ctx, cross_k, 0, 1, 3, 2), 0, 2, 1, 3));

            ggml_tensor * cross_v =
                ggml_cont(ctx, ggml_view_4d(ctx, cache->cross_v_l[l], model->max_encoder_context_length,
                                            model->head_size, model->decoder_attn_heads, 2,
                                            model->max_encoder_context_length * ggml_element_size(cache->cross_v_l[l]),
                                            model->head_size * model->max_encoder_context_length *
                                                ggml_element_size(cache->cross_v_l[l]),
                                            model->head_size * model->max_encoder_context_length *
                                                model->decoder_attn_heads * ggml_element_size(cache->cross_v_l[l]),
                                            0));

            // As noted in the encoder Dia uses the Neo-X protocol for RoPE.
            cross_Qcur             = ggml_rope(ctx,
                                               ggml_cont(ctx, ggml_reshape_4d(ctx, cross_Qcur, model->head_size,
                                                                              model->decoder_attn_heads, batch.sequence_length, 2)),
                                               dctx->positions, model->head_size, 2);
            ggml_tensor * cross_q  = ggml_cont(ctx, ggml_permute(ctx, cross_Qcur, 0, 2, 1, 3));
            ggml_tensor * cross_kq = ggml_mul_mat(ctx, cross_k, cross_q);

            // given that attention bias, scaling and masking are not used for decoding, it might be faster to prefer the #ggml_soft_max op here,
            cross_kq                       = ggml_soft_max_ext(ctx, cross_kq, nullptr, 1.0f, 0.0f);
            ggml_tensor * cross_kqv        = ggml_mul_mat(ctx, cross_kq, cross_v);
            ggml_tensor * cross_kqv_merged = ggml_cont(ctx, ggml_permute(ctx, cross_kqv, 2, 0, 1, 3));
            cur = ggml_cont_3d(ctx, cross_kqv_merged, model->decoder_hidden_size, batch.sequence_length, 2);
            cur = ggml_mul_mat(ctx, layer->cross_attn_o, cur);
        }

        // if we ever need to support multiple step decoder runs then this reshape will need to be replaced with permutation.
        cur                        = ggml_cont_2d(ctx, cur, cur->ne[0], 2);
        cur                        = ggml_add(ctx, cur, residual_cross);
        ggml_tensor * residual_mlp = cur;

        cur = dia_layer_norm(ctx, cur, layer->mlp_norm);
        // mlp
        {
            cur = ggml_mul(ctx, ggml_silu(ctx, ggml_mul_mat(ctx, layer->gate, cur)), ggml_mul_mat(ctx, layer->up, cur));
            cur = ggml_mul_mat(ctx, layer->out, cur);
        }

        cur = ggml_add(ctx, cur, residual_mlp);
    }

    cur = dia_layer_norm(ctx, cur, model->decoder->norm);
    cur = build_dia_head_outputs(ctx, model, cur);
    return cur;
}
