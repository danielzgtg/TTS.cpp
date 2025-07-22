
ggml_tensor * parler_build_inp_embd(ggml_context * ctx, parler_context * pctx, parler_tts_model * model,
                                    parler_ubatch & batch) {
    // Parler has two embedding schemas one for the text input and one for generative audio tokens. These two schemas have effectively distinct shapes (i.e. [batch_size, sequence_length] and [batch_size, sequence_lenghth, num_codebooks] respectively).
    // This means that depending on where we are in generation we need to follow a distinct pattern
    ggml_tensor * input_embs;
    pctx->positions = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, batch.sequence_length);
    ggml_set_input(pctx->positions);
    if (batch.audio_generation) {
        pctx->audio_inp_tokens = ggml_reshape_2d(ctx, ggml_new_tensor_1d(ctx, GGML_TYPE_I32, batch.n_audio_tokens),
                                                 batch.n_audio_tokens / model->n_output_heads, model->n_output_heads);
        ggml_set_input(pctx->audio_inp_tokens);
        ggml_tensor * audio_tokens = ggml_reshape_2d(
            ctx, pctx->audio_inp_tokens, batch.n_audio_tokens / model->n_output_heads, model->n_output_heads);
        for (int i = 0; i < model->n_output_heads; i++) {
            if (i == 0) {
                input_embs =
                    ggml_get_rows(ctx, model->embds[i],
                                  ggml_view_2d(ctx, audio_tokens, 1, batch.n_audio_tokens / model->n_output_heads,
                                               audio_tokens->nb[1], i * sizeof(int32_t)));
            } else {
                input_embs = ggml_add(
                    ctx,
                    ggml_get_rows(ctx, model->embds[i],
                                  ggml_view_2d(ctx, audio_tokens, 1, batch.n_audio_tokens / model->n_output_heads,
                                               audio_tokens->nb[1], i * sizeof(int32_t))),
                    input_embs);
            }
        }
    } else {
        pctx->inp_tokens = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, batch.n_tokens);
        ggml_set_input(pctx->inp_tokens);
        input_embs = ggml_get_rows(ctx, model->prompt_embd, pctx->inp_tokens);
    }
    return ggml_add(ctx, input_embs, ggml_get_rows(ctx, model->precomputed_positional_embds, pctx->positions));
}

ggml_tensor * parler_build_layer_norm(ggml_context * ctx, ggml_tensor * inputs, ggml_tensor * weight,
                                      ggml_tensor * bias) {
    // parler always uses default eps
    float eps = 0.00001;
    inputs    = ggml_norm(ctx, inputs, eps);
    inputs    = ggml_mul(ctx, inputs, weight);
    return ggml_add(ctx, inputs, bias);
}

void parler_build_kv_store(ggml_context * ctx, parler_kv_cache * kv, ggml_cgraph * graph, ggml_tensor * k_cur,
                           ggml_tensor * v_cur, int32_t n_tokens, int32_t kv_head, int32_t index, int32_t n_embd_gqa) {
    // this is the max context size;
    const int64_t n_ctx = 4096;

    ggml_tensor * k_cache_view = ggml_view_1d(ctx, kv->k_l[index], n_tokens * n_embd_gqa,
                                              ggml_row_size(kv->k_l[index]->type, n_embd_gqa) * kv_head);

    ggml_build_forward_expand(graph, ggml_cpy(ctx, k_cur, k_cache_view));

    assert(v_cur->ne[0] == n_embd_gqa && v_cur->ne[1] == n_tokens);

    ggml_tensor * v_cache_view = nullptr;

    v_cache_view = ggml_view_2d(ctx, kv->v_l[index], n_tokens, n_embd_gqa, (n_ctx) *ggml_element_size(kv->v_l[index]),
                                (kv_head) *ggml_element_size(kv->v_l[index]));

    v_cur = ggml_cont(ctx, ggml_transpose(ctx, v_cur));

    ggml_build_forward_expand(graph, ggml_cpy(ctx, v_cur, v_cache_view));
}

ggml_tensor * parler_build_head_outputs(ggml_context * ctx, parler_tts_model * model, ggml_tensor * cur) {
    // going to cat the heads together and then reshape them;
    // honestly ggml doesn't provide good support for stacking and discrete tensor access
    ggml_tensor * out;
    for (int i = 0; i < model->n_output_heads; i++) {
        if (i == 0) {
            out = ggml_mul_mat(ctx, model->heads[i], cur);
        } else {
            out = ggml_concat(ctx, out, ggml_mul_mat(ctx, model->heads[i], cur), 1);
        }
    }
    ggml_set_name(out, "final_out");
    //out = ggml_cont(ctx, ggml_transpose(ctx, out));

    int32_t sql_len = (int32_t) (ggml_nelements(out) / (model->output_vocab_size * model->n_output_heads));
    return ggml_cont_3d(ctx, out, model->output_vocab_size, sql_len, model->n_output_heads);
}

ggml_tensor * build_attn_mask(ggml_context * ctx, parler_context * pctx, parler_ubatch & batch) {
    pctx->attn_mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, (int64_t) pctx->current_position + batch.sequence_length,
                                         (int64_t) pctx->current_position + batch.sequence_length);
    ggml_set_input(pctx->attn_mask);

    return pctx->attn_mask;
}

ggml_tensor * build_attn_mask_cross(ggml_context * ctx, parler_context * pctx, parler_tts_model * model,
                                    parler_ubatch & batch) {
    pctx->attn_mask_cross =
        ggml_new_tensor_2d(ctx, GGML_TYPE_F32, (int64_t) model->n_encode_length, (int64_t) batch.sequence_length);
    ggml_set_input(pctx->attn_mask_cross);

    return pctx->attn_mask_cross;
}

void parler_tts_runner::prepare_post_load() {
    if (config.use_cross_attn) {
        runner->model->prep_cross_key_values(n_threads);
    }
    dac_runner->prepare_post_load();
    parler_kv_cache_init(kv_self, model, pctx, std::mt19937(std::random_device{}())());
    auto batch = build_worst_case_batch();
    auto gf    = build_parler_graph(batch);
    pctx->prep_schedule(gf);
}

parler_ubatch parler_tts_runner::build_worst_case_batch() {
    parler_ubatch batch;
    batch.audio_generation = false;
    batch.n_tokens         = model->max_ctx_length;
    batch.n_audio_tokens   = 0;
    batch.sequence_length  = model->max_ctx_length;
    return batch;
}


ggml_cgraph * parler_tts_runner::build_parler_graph(parler_ubatch & batch) {
    init_build();
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 8192, false);

    ggml_tensor * cur;
    ggml_tensor * inpL;

    const int32_t full_sequence_length = pctx->current_position + (uint32_t) batch.sequence_length;

    inpL = parler_build_inp_embd(ctx, pctx, model, batch);

    ggml_tensor * KQ_mask_dec = build_attn_mask(ctx, pctx, batch);
    ggml_tensor * KQ_mask_cross = build_attn_mask_cross(ctx, pctx, model, batch);

    for (int l = 0; l < model->n_layers; l++) {
        ggml_tensor * residual = inpL;
        ggml_set_name(inpL, ("layer_" + std::to_string(l) + "_input").c_str());

        cur = parler_build_layer_norm(ctx, inpL, model->layers[l]->self_attn_norm, model->layers[l]->self_attn_norm_bias);

        ggml_tensor * attn_out;

        // self-attention
        {
            ggml_tensor * Qcur = ggml_mul_mat(ctx, model->layers[l]->self_attn_q_proj, cur);
            ggml_tensor * Kcur = ggml_mul_mat(ctx, model->layers[l]->self_attn_k_proj, cur);
            ggml_tensor * Vcur = ggml_mul_mat(ctx, model->layers[l]->self_attn_v_proj, cur);

            parler_build_kv_store(ctx, kv_self, gf, Kcur, Vcur, (int32_t) batch.sequence_length, pctx->current_position, l, model->hidden_size);
            ggml_tensor * k =
                ggml_view_3d(ctx, kv_self->k_l[l],
                        model->head_size, full_sequence_length, model->n_attn_heads,
                        ggml_row_size(kv_self->k_l[l]->type, model->hidden_size),
                        ggml_row_size(kv_self->k_l[l]->type, model->head_size),
                        0);


            ggml_tensor * v =
                ggml_view_3d(ctx, kv_self->v_l[l],
                        full_sequence_length, model->head_size, model->n_attn_heads,
                        ggml_element_size(kv_self->v_l[l])*model->max_ctx_length,
                        ggml_element_size(kv_self->v_l[l])*model->max_ctx_length*model->head_size,
                        0);

            Qcur = ggml_reshape_3d(ctx, Qcur, model->head_size, model->n_attn_heads, batch.sequence_length);
            ggml_tensor * q = ggml_cont(ctx, ggml_permute(ctx, Qcur, 0, 2, 1, 3));
            ggml_tensor * kq = ggml_mul_mat(ctx, ggml_cont(ctx, k), q);
            kq = ggml_soft_max_ext(ctx, kq, KQ_mask_dec, 1.0f/sqrtf(model->head_size), 0.0f);
            ggml_tensor * kqv = ggml_mul_mat(ctx, kq, v);
            ggml_tensor * kqv_merged = ggml_permute(ctx, kqv, 2, 0, 1, 3);
            attn_out = ggml_cont_2d(ctx, kqv_merged, model->hidden_size, batch.sequence_length);
            attn_out = ggml_mul_mat(ctx, model->layers[l]->self_attn_o_proj, attn_out);
        }

        cur = ggml_add(ctx, attn_out, residual);

        if (model->use_cross_attn) {
            ggml_tensor * residuala = cur;

            // norm
            cur = parler_build_layer_norm(ctx, cur, model->layers[l]->attn_norm, model->layers[l]->attn_norm_bias);

            //cross-attention
            ggml_tensor * Qcur = ggml_mul_mat(ctx, model->layers[l]->attn_q_proj, cur);
            Qcur = ggml_reshape_3d(ctx, Qcur, model->head_size, model->n_attn_heads, batch.sequence_length);

            ggml_tensor * q = ggml_cont(ctx, ggml_permute(ctx, Qcur, 0, 2, 1, 3));

            ggml_tensor * kq = ggml_mul_mat(ctx, model->layers[l]->cross_k, q);
            kq = ggml_soft_max_ext(ctx, kq, KQ_mask_cross, 1.0f/sqrtf(model->head_size), 0.0f);

            ggml_tensor * kqv  = ggml_mul_mat(ctx, kq, model->layers[l]->cross_v);
            ggml_tensor * kqv_merged = ggml_permute(ctx, kqv, 2, 0, 1, 3);
            cur = ggml_cont_2d(ctx, kqv_merged, model->hidden_size, batch.sequence_length);
            cur = ggml_mul_mat(ctx, model->layers[l]->attn_o_proj, cur);
            cur = ggml_add(ctx, cur, residuala);
        }

        ggml_tensor * residualffn = cur;

        cur = parler_build_layer_norm(ctx, cur, model->layers[l]->final_norm, model->layers[l]->final_norm_bias);
        cur = ggml_mul_mat(ctx, model->layers[l]->fc1, cur);
        cur = ggml_gelu(ctx, cur);
        cur = ggml_mul_mat(ctx, model->layers[l]->fc2, cur);
        cur = ggml_add(ctx, cur, residualffn);
        inpL = cur;
    }

    cur = parler_build_layer_norm(ctx, cur, model->layer_norm, model->layer_norm_bias);
    cur = parler_build_head_outputs(ctx, model, cur);
    ggml_build_forward_expand(gf, cur);
    free_build();

    return gf;
}
