
static ggml_tensor * build_dia_encoder_attn_mask(ggml_context * ctx, dia_context * dctx, dia_model * model) {
    dctx->encode_attn_mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, (int64_t) model->max_encoder_context_length,
                                                (int64_t) model->max_encoder_context_length);
    ggml_set_input(dctx->encode_attn_mask);

    return dctx->encode_attn_mask;
}


static ggml_tensor * build_dia_encoder(ggml_context * ctx, dia_model * model, dia_context * dctx, dia_ubatch & batch) {
    dctx->inp_tokens = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, model->max_encoder_context_length * 2);
    ggml_set_input(dctx->inp_tokens);

    dctx->encode_positions = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, model->max_encoder_context_length);
    ggml_set_input(dctx->encode_positions);

    ggml_tensor * attn_mask = build_dia_encoder_attn_mask(ctx, dctx, model);

    ggml_tensor * cur = ggml_reshape_3d(ctx, ggml_get_rows(ctx, model->encoder->embedding, dctx->inp_tokens),
                                        model->encoder_hidden_size, model->max_encoder_context_length, 2);
    for (auto layer : model->encoder->layers) {
        ggml_tensor * residual = cur;

        cur = dia_layer_norm(ctx, cur, layer->self_attn_norm);
        // self-attention
        {
            ggml_tensor * Qcur = ggml_mul_mat(ctx, layer->q, cur);
            ggml_tensor * Kcur = ggml_mul_mat(ctx, layer->k, cur);
            ggml_tensor * Vcur = ggml_mul_mat(ctx, layer->v, cur);

            // Strangely Dia follows the neoX Rotary Positional Embeddings Protocol
            Qcur                     = ggml_rope(ctx,
                                                 ggml_cont(ctx, ggml_reshape_4d(ctx, Qcur, model->head_size, model->encoder_attn_heads,
                                                                                model->max_encoder_context_length, 2)),
                                                 dctx->encode_positions, model->head_size, 2);
            Kcur                     = ggml_rope(ctx,
                                                 ggml_cont(ctx, ggml_reshape_4d(ctx, Kcur, model->head_size, model->encoder_attn_heads,
                                                                                model->max_encoder_context_length, 2)),
                                                 dctx->encode_positions, model->head_size, 2);
            ggml_tensor * q          = ggml_cont(ctx, ggml_permute(ctx, Qcur, 0, 2, 1, 3));
            ggml_tensor * k          = ggml_cont(ctx, ggml_permute(ctx, Kcur, 0, 2, 1, 3));
            ggml_tensor * kq         = ggml_mul_mat(ctx, k, q);
            kq                       = ggml_soft_max_ext(ctx, kq, attn_mask, 1.0f, 0.0f);
            ggml_tensor * v          = ggml_cont_4d(ctx, ggml_transpose(ctx, Vcur), model->max_encoder_context_length,
                                                    model->head_size, model->encoder_attn_heads, 2);
            ggml_tensor * kqv        = ggml_mul_mat(ctx, kq, v);
            ggml_tensor * kqv_merged = ggml_permute(ctx, kqv, 2, 0, 1, 3);

            // It is unclear why the attention ops in Dia's encoder don't project to the embedding dimension size as is standard. Instead they up project to the decoder's embedding dimension
            // then down project back the the encoder embedding dimension.
            cur = ggml_cont_3d(ctx, kqv_merged, model->decoder_hidden_size, model->max_encoder_context_length, 2);
            cur = ggml_mul_mat(ctx, layer->o, cur);
        }

        cur                        = ggml_add(ctx, cur, residual);
        ggml_tensor * residual_mlp = cur;

        cur = dia_layer_norm(ctx, cur, layer->mlp_norm);
        // mlp
        {
            cur = ggml_mul(ctx, ggml_silu(ctx, ggml_mul_mat(ctx, layer->gate, cur)), ggml_mul_mat(ctx, layer->up, cur));
            cur = ggml_mul_mat(ctx, layer->out, cur);
        }

        cur = ggml_add(ctx, cur, residual_mlp);
    }

    cur = dia_layer_norm(ctx, cur, model->encoder->norm);
    return cur;
}
