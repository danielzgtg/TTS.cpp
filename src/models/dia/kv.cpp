
static bool dia_kv_cache_init(dia_kv_cache * cache, dia_model * model, dia_context * dctx) {
    ggml_backend_buffer_type_t buft = nullptr;
    // this will only really support cpu or metal for the time being;
    if (dctx->backend != nullptr) {
#ifdef GGML_USE_METAL
        buft = ggml_backend_metal_buffer_type();
#endif
    } else {
        buft = ggml_backend_cpu_buffer_type();
    }

    ggml_init_params params = {
        /*.mem_size   =*/(4u * model->n_decoder_layers + 1) * ggml_tensor_overhead(),
        /*.mem_buffer =*/NULL,
        /*.no_alloc   =*/true,
    };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        return false;
    }
    cache->ctx = ctx;

    cache->k_l.reserve(model->n_decoder_layers);
    cache->v_l.reserve(model->n_decoder_layers);
    cache->cross_k_l.reserve(model->n_decoder_layers);
    cache->cross_v_l.reserve(model->n_decoder_layers);

    for (int i = 0; i < (int) model->n_decoder_layers; i++) {
        ggml_tensor * k =
            ggml_new_tensor_1d(cache->ctx, cache->tensor_type,
                               model->head_size * model->decoder_attn_heads * model->max_generation_size * 2);
        ggml_tensor * v =
            ggml_new_tensor_1d(cache->ctx, cache->tensor_type,
                               model->head_size * model->decoder_attn_heads * model->max_generation_size * 2);
        ggml_tensor * cross_k =
            ggml_new_tensor_1d(cache->ctx, cache->tensor_type,
                               model->head_size * model->decoder_attn_heads * model->max_encoder_context_length * 2);
        ggml_tensor * cross_v =
            ggml_new_tensor_1d(cache->ctx, cache->tensor_type,
                               model->head_size * model->decoder_attn_heads * model->max_encoder_context_length * 2);
        ggml_format_name(k, "cache_k_l%d", i);
        ggml_format_name(v, "cache_v_l%d", i);
        ggml_format_name(cross_k, "cache_cross_k_l%d", i);
        ggml_format_name(cross_v, "cache_cross_v_l%d", i);
        cache->k_l.push_back(k);
        cache->v_l.push_back(v);
        cache->cross_k_l.push_back(cross_k);
        cache->cross_v_l.push_back(cross_v);
    }

    // allocate tensors and initialize the buffers to avoid NaNs in the padding
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(cache->ctx, buft);
    if (!buf) {
        return false;
    }
    ggml_backend_buffer_clear(buf, 0);
    cache->buf = buf;

    return true;
}
