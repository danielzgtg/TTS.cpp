
static ggml_tensor * build_t5_norm(ggml_context * ctx, ggml_tensor * cur, ggml_tensor * weight) {
    // this is static for all versions of T5 FLAN
    float eps = 0.000001;
    cur       = ggml_rms_norm(ctx, cur, eps);
    cur       = ggml_mul(ctx, cur, weight);
    return cur;
}

static ggml_tensor * build_t5_attn_mask(ggml_context * ctx, t5_context * t5ctx, const t5_ubatch & batch) {
    t5ctx->attn_mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, (int64_t) batch.n_tokens, (int64_t) batch.n_tokens);
    ggml_set_input(t5ctx->attn_mask);

    return t5ctx->attn_mask;
}

static ggml_tensor * build_t5_pos_bias(ggml_context * ctx, ggml_tensor * pos_bucket, ggml_tensor * relative_attn_bias) {
    ggml_tensor * pos_bucket_1d = ggml_view_1d(ctx, pos_bucket, pos_bucket->ne[0] * pos_bucket->ne[1], 0);
    ggml_tensor * pos_bias      = ggml_get_rows(ctx, relative_attn_bias, pos_bucket_1d);

    pos_bias = ggml_view_3d(ctx, pos_bias, pos_bias->ne[0], pos_bucket->ne[0], pos_bucket->ne[1],
                            ggml_element_size(pos_bias) * pos_bias->ne[0],
                            ggml_element_size(pos_bias) * pos_bias->ne[0] * pos_bucket->ne[0], 0);
    pos_bias = ggml_permute(ctx, pos_bias, 2, 1, 0, 3);
    pos_bias = ggml_cont(ctx, pos_bias);
    return pos_bias;
}

t5_ubatch t5_runner::build_worst_case_batch() {
    t5_ubatch batch;
    batch.n_tokens = model->max_context_length;
    return batch;
}

void t5_runner::prepare_post_load() {
    auto batch = build_worst_case_batch();
    auto gf    = build_t5_graph(batch);
    t5ctx->prep_schedule(gf);
}

ggml_cgraph * t5_runner::build_t5_graph(t5_ubatch & batch) {
    init_build();
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 8192, false);

    ggml_tensor * cur;
    ggml_tensor * inpL;

    //t5ctx->positions = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, batch.n_tokens);
    //ggml_set_input(t5ctx->positions);

    t5ctx->inp_pos_bucket = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, batch.n_tokens, batch.n_tokens);
    ggml_set_input(t5ctx->inp_pos_bucket);

    t5ctx->inp_tokens = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, batch.n_tokens);
    ggml_set_input(t5ctx->inp_tokens);

    inpL = ggml_get_rows(ctx, model->embd, t5ctx->inp_tokens);

    ggml_tensor * KQ_mask_dec = build_t5_attn_mask(ctx, t5ctx, batch);
    ggml_tensor * pos_bias    = build_t5_pos_bias(ctx, t5ctx->inp_pos_bucket, model->relative_attn_bias);

    for (int l = 0; l < model->n_layers; l++) {
        ggml_tensor * residual = inpL;

        cur = build_t5_norm(ctx, inpL, model->layers[l].attn_norm);

        ggml_tensor * attn_out;

        // self-attention
        {
            ggml_tensor * Qcur = ggml_mul_mat(ctx, model->layers[l].q, cur);
            ggml_tensor * Kcur = ggml_mul_mat(ctx, model->layers[l].k, cur);
            ggml_tensor * Vcur = ggml_mul_mat(ctx, model->layers[l].v, cur);

            Qcur = ggml_reshape_3d(ctx, Qcur, model->head_size, model->n_attn_heads, batch.n_tokens);
            Kcur = ggml_reshape_3d(ctx, Kcur, model->head_size, model->n_attn_heads, batch.n_tokens);

            ggml_tensor * q = ggml_permute(ctx, Qcur, 0, 2, 1, 3);
            ggml_tensor * k = ggml_cont(ctx, ggml_permute(ctx, Kcur, 0, 2, 1, 3));

            ggml_tensor * kq = ggml_mul_mat(ctx, k, q);
            kq               = ggml_add(ctx, kq, pos_bias);

            kq = ggml_soft_max_ext(ctx, kq, KQ_mask_dec, 1.0f, 0.0f);

            ggml_tensor * v =
                ggml_cont_3d(ctx, ggml_transpose(ctx, Vcur), batch.n_tokens, model->head_size, model->n_attn_heads);
            ggml_tensor * kqv        = ggml_mul_mat(ctx, kq, v);
            ggml_tensor * kqv_merged = ggml_permute(ctx, kqv, 2, 0, 1, 3);
            attn_out                 = ggml_cont_2d(ctx, kqv_merged, model->hidden_size, batch.n_tokens);
            attn_out                 = ggml_mul_mat(ctx, model->layers[l].o, attn_out);
        }

        cur                       = ggml_add(ctx, attn_out, residual);
        ggml_tensor * residualmlp = cur;

        // mlp
        {
            cur                     = build_t5_norm(ctx, cur, model->layers[l].mlp_norm);
            ggml_tensor * gate_proj = ggml_mul_mat(ctx, model->layers[l].wi_1, cur);
            cur = ggml_mul(ctx, ggml_gelu(ctx, ggml_mul_mat(ctx, model->layers[l].wi_0, cur)), gate_proj);
            cur = ggml_mul_mat(ctx, model->layers[l].wo, cur);
        }

        cur  = ggml_add(ctx, cur, residualmlp);
        inpL = cur;
    }

    cur = build_t5_norm(ctx, cur, model->out_norm);

    if (model->down_proj) {
        cur = ggml_mul_mat(ctx, model->down_proj, cur);
    }

    if (model->down_proj_bias) {
        cur = ggml_add(ctx, cur, model->down_proj_bias);
    }

    ggml_build_forward_expand(gf, cur);

    free_build();

    return gf;
}
