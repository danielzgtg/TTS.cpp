
static ggml_tensor * repeat_interleave_dim1(ggml_context * ctx, ggml_tensor * a, int repeat) {
    //return ggml_repeat(ctx, a, ggml_new_tensor_4d(ctx, GGML_TYPE_F32, a->ne[0], 4*a->ne[1], a->ne[2], a->ne[3]));
    ggml_tensor * running;
    for (int i = 0; i < a->ne[1]; i++) {
        int           offset = i * a->nb[1];
        ggml_tensor * t =
            ggml_cont(ctx, ggml_view_4d(ctx, a, a->ne[0], 1, a->ne[2], a->ne[3], a->nb[1], a->nb[2], a->nb[3], offset));
        t = ggml_repeat(ctx, t, ggml_new_tensor_4d(ctx, GGML_TYPE_F32, a->ne[0], repeat, a->ne[2], a->ne[3]));
        if (i == 0) {
            running = t;
        } else {
            running = ggml_concat(ctx, running, t, 1);
        }
    }
    return running;
}

static void build_dia_self_kv_store(ggml_context * ctx, dia_context * dctx, dia_model * model, dia_kv_cache * kv,
                                    ggml_cgraph * gf, ggml_tensor * k, ggml_tensor * v, dia_ubatch & batch,
                                    int layer_index) {
    int64_t attn_size = model->head_size * model->decoder_attn_heads;

    ggml_tensor * k_cache_view =
        ggml_view_2d(ctx, kv->k_l[layer_index], attn_size, 2,
                     attn_size * model->max_generation_size * ggml_element_size(kv->k_l[layer_index]),
                     attn_size * dctx->current_position * ggml_element_size(kv->k_l[layer_index]));

    k = ggml_rope(
        ctx,
        ggml_cont(ctx, ggml_reshape_4d(ctx, k, model->head_size, model->decoder_attn_heads / model->decoder_query_heads,
                                       batch.sequence_length, 2)),
        dctx->positions, model->head_size, 2);
    // Since the sequence length should always be 1 here this is the most pertinent time to repeat the heads for grouped query attention.
    // If GGML supported a repeat_interleave op then it would be more optimal to store just the groups in the cache and interleave the attention heads after recalling
    // from the cache
    k = repeat_interleave_dim1(
        ctx,
        ggml_cont(ctx, ggml_reshape_4d(ctx, k, model->head_size, model->decoder_attn_heads / model->decoder_query_heads,
                                       batch.sequence_length, 2)),
        model->decoder_query_heads);
    k = ggml_cont(ctx, ggml_reshape_2d(ctx, k, attn_size, 2));

    ggml_build_forward_expand(gf, ggml_cpy(ctx, k, k_cache_view));

    ggml_tensor * v_cache_view = nullptr;

    v_cache_view = ggml_view_2d(ctx, kv->v_l[layer_index], attn_size, 2,
                                attn_size * model->max_generation_size * ggml_element_size(kv->v_l[layer_index]),
                                attn_size * dctx->current_position * ggml_element_size(kv->v_l[layer_index]));

    // Since the sequence length should always be 1 here this is the most pertinent time to repeat the heads for grouped query attention.
    // If GGML supported a repeat_interleave op then it would be more optimal to store just the groups in the cache and interleave the attention heads after recalling
    // from the cache
    v = repeat_interleave_dim1(
        ctx,
        ggml_cont(ctx, ggml_reshape_4d(ctx, v, model->head_size, model->decoder_attn_heads / model->decoder_query_heads,
                                       batch.sequence_length, 2)),
        model->decoder_query_heads);

    ggml_build_forward_expand(gf, ggml_cpy(ctx, v, v_cache_view));
}
