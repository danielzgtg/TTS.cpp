
static void build_dia_cross_kv_store(ggml_context * ctx, dia_context * dctx, dia_model * model, dia_kv_cache * kv,
                                     ggml_cgraph * gf, ggml_tensor * encoder_hidden_states, int layer_index) {
    dia_decoder_layer * layer = model->decoder->layers[layer_index];
    ggml_tensor *       encoder_states_key_view =
        ggml_cont(ctx, ggml_view_3d(ctx, encoder_hidden_states, model->encoder_hidden_size, dctx->prompt_size, 2,
                                    model->encoder_hidden_size * ggml_element_size(encoder_hidden_states),
                                    model->encoder_hidden_size * model->max_encoder_context_length *
                                        ggml_element_size(encoder_hidden_states),
                                    0));

    ggml_tensor * k              = ggml_mul_mat(ctx, layer->cross_attn_k, encoder_states_key_view);
    ggml_tensor * positions_view = ggml_view_1d(ctx, dctx->encode_positions, dctx->prompt_size, 0);

    k = ggml_rope(
        ctx, ggml_cont(ctx, ggml_reshape_4d(ctx, k, model->head_size, model->decoder_attn_heads, dctx->prompt_size, 2)),
        positions_view, model->head_size, 2);
    k = ggml_cont(ctx, ggml_permute(ctx, k, 0, 1, 3, 2));

    ggml_tensor * k_cache_view = ggml_view_4d(
        ctx, kv->cross_k_l[layer_index], model->head_size, model->decoder_attn_heads, 2, dctx->prompt_size,
        model->head_size * ggml_element_size(kv->cross_k_l[layer_index]),
        model->head_size * model->decoder_attn_heads * ggml_element_size(kv->cross_k_l[layer_index]),
        model->head_size * model->decoder_attn_heads * 2 * ggml_element_size(kv->cross_k_l[layer_index]), 0);

    ggml_build_forward_expand(gf, ggml_cpy(ctx, k, k_cache_view));

    ggml_tensor * v =
        ggml_cont(ctx, ggml_transpose(ctx, ggml_mul_mat(ctx, layer->cross_attn_v, encoder_hidden_states)));
    v = ggml_cont_4d(ctx, v, model->max_encoder_context_length, model->head_size, model->decoder_attn_heads, 2);

    ggml_tensor * v_cache_view = ggml_view_4d(
        ctx, kv->cross_v_l[layer_index], model->max_encoder_context_length, model->head_size, model->decoder_attn_heads,
        2, model->max_encoder_context_length * ggml_element_size(kv->cross_v_l[layer_index]),
        model->head_size * model->max_encoder_context_length * ggml_element_size(kv->cross_v_l[layer_index]),
        model->head_size * model->max_encoder_context_length * model->decoder_attn_heads *
            ggml_element_size(kv->cross_v_l[layer_index]),
        0);

    ggml_build_forward_expand(gf, ggml_cpy(ctx, v, v_cache_view));
}
