#include "kv.h"

void parler_tts_model::prep_cross_key_values(int n_threads, tts_response * conditional_prompt) {
    ggml_backend_t backend_cpu = ggml_backend_cpu_init();
    ggml_backend_buffer_type_t backend_cpu_buffer = ggml_backend_cpu_buffer_type();
    // Let it create a disposable threadpool just this once
    ggml_backend_cpu_set_n_threads(backend_cpu, n_threads);
    std::vector<ggml_backend_buffer_type_t> bufs = {backend_cpu_buffer};
    std::vector<ggml_backend_t> backs = {backend_cpu};
    ggml_backend_sched_t sched = ggml_backend_sched_new(backs.data(), bufs.data(), 1, max_cross_nodes*n_layers, false);

    std::vector<uint8_t> buf_compute_meta;
    buf_compute_meta.resize(max_cross_nodes*n_layers*ggml_tensor_overhead() + ggml_graph_overhead_custom(max_cross_nodes*n_layers, false));

    ggml_init_params params = {
        /*.mem_size   =*/ buf_compute_meta.size(),
        /*.mem_buffer =*/ buf_compute_meta.data(),
        /*.no_alloc   =*/ true,
    };
    ggml_context * cctx = ggml_init(params);
    ggml_cgraph * gf = ggml_new_graph_custom(cctx, 4096, false);
    if (conditional_prompt) {
        // If we are updating the conditional prompt then we have to reset the tensor offsets into the ggml_context otherwise we could overflow the assigned buffer and lose our prompt.
        // These offsets are assigned by #set_tensor below.
        offset -= n_encode_length*hidden_size*sizeof(float)*n_layers*2;
        precomputed_input_emb = ggml_new_tensor_2d(cctx, GGML_TYPE_F32, conditional_prompt->hidden_size, conditional_prompt->n_outputs);
        ggml_set_input(precomputed_input_emb);
        n_encode_length = conditional_prompt->n_outputs;
    }

    for (int i = 0; i < layers.size(); i++) {
        ggml_tensor * Kcur = ggml_mul_mat(cctx, layers[i]->attn_k_proj, precomputed_input_emb);
        ggml_tensor * Vcur = ggml_mul_mat(cctx, layers[i]->attn_v_proj, precomputed_input_emb);

        Kcur = ggml_reshape_3d(cctx, Kcur, head_size, n_attn_heads, n_encode_length);
        Vcur = ggml_transpose(cctx, Vcur);

        ggml_tensor * k = ggml_cont(cctx, ggml_permute(cctx, Kcur, 0, 2, 1, 3));
        ggml_set_name(k, ("cross_key_" + std::to_string(i)).c_str());
        ggml_build_forward_expand(gf, k);

        ggml_tensor * v = ggml_cont_3d(cctx, Vcur, n_encode_length, head_size, n_attn_heads);
        ggml_set_name(v, ("cross_value_" + std::to_string(i)).c_str());
        ggml_build_forward_expand(gf, v);
    }

    ggml_free(cctx);
    ggml_backend_sched_reserve(sched, gf);
    ggml_backend_sched_alloc_graph(sched, gf);
    if (conditional_prompt) {
        ggml_backend_tensor_set(precomputed_input_emb, conditional_prompt->data, 0, conditional_prompt->n_outputs*conditional_prompt->hidden_size*ggml_element_size(precomputed_input_emb));
    }

    ggml_backend_sched_graph_compute_async(sched, gf);

    for (int i = 0; i < layers.size(); i++) {
        ggml_tensor * k = ggml_graph_get_tensor(gf, ("cross_key_" + std::to_string(i)).c_str());
        layers[i]->cross_k = ggml_dup_tensor(ctx, k);
        set_tensor(layers[i]->cross_k, k);
        ggml_tensor * v = ggml_graph_get_tensor(gf, ("cross_value_" + std::to_string(i)).c_str());
        layers[i]->cross_v = ggml_dup_tensor(ctx, v);
        set_tensor(layers[i]->cross_v, v);
    }
    ggml_backend_sched_free(sched);
    ggml_backend_free(backend_cpu);
}


static bool parler_kv_cache_init(parler_kv_cache * cache, parler_tts_model * model, parler_context * pctx, int32_t seq_id) {
    const int64_t n_layer = (int64_t) model->layers.size();
    cache->seq_id = seq_id;

    ggml_backend_buffer_type_t buft = nullptr;
    // this will only really support cpu or metal for the time being;
    if (pctx->backend != nullptr) {
#ifdef GGML_USE_METAL
        buft = ggml_backend_metal_buffer_type();
#endif
    } else {
        buft = ggml_backend_cpu_buffer_type();
    }

    ggml_init_params params = {
        /*.mem_size   =*/ (2u*model->n_layers+1)*ggml_tensor_overhead(),
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        return false;
    }
    cache->ctx = ctx;


    cache->k_l.reserve(n_layer);
    cache->v_l.reserve(n_layer);

    for (int i = 0; i < (int) n_layer; i++) {
        ggml_tensor * k = ggml_new_tensor_1d(cache->ctx, cache->type_k, model->hidden_size*model->max_ctx_length);
        ggml_tensor * v = ggml_new_tensor_1d(cache->ctx, cache->type_v, model->hidden_size*model->max_ctx_length);
        ggml_format_name(k, "cache_k_l%d", i);
        ggml_format_name(v, "cache_v_l%d", i);
        cache->k_l.push_back(k);
        cache->v_l.push_back(v);
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
