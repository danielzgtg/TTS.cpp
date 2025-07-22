
struct t5_context * build_new_t5_context(struct t5_encoder * model, int n_threads, bool use_cpu) {
    t5_context * t5ctx = new t5_context(model, n_threads);
    if (!use_cpu) {
#ifdef GGML_USE_METAL
        t5ctx->backend = ggml_backend_metal_init();
#endif
    }
    t5ctx->backend_cpu = ggml_backend_cpu_init();
    t5ctx->set_threads();
    t5ctx->build_schedule();
    t5ctx->buf_compute_meta.resize(ggml_tensor_overhead()*model->max_nodes() + ggml_graph_overhead_custom(model->max_nodes(), false));
    return t5ctx;
}


void t5_runner::set_inputs(t5_ubatch & batch) {
    ggml_backend_tensor_set(t5ctx->inp_tokens, batch.input_tokens, 0,
                            batch.n_tokens * ggml_element_size(t5ctx->inp_tokens));
    float *    attn_mask          = nullptr;
    uint32_t * positions          = nullptr;
    uint32_t * pos_bucket         = nullptr;
    attn_mask                     = (float *) t5ctx->attn_mask->data;
    positions                     = (uint32_t *) t5ctx->positions->data;
    pos_bucket                    = (uint32_t *) t5ctx->inp_pos_bucket->data;
    int   n_buckets               = (int) model->relative_attn_buckets / 2;
    int   max_exact               = (int) n_buckets / 2;
    float logarithmic_denominator = log(128.0 / max_exact);
    for (int i = 0; i < batch.n_tokens; i++) {
        for (int ii = 0; ii < batch.n_tokens; ii++) {
            int ab_rpos                        = abs(i - ii);
            int rpos                           = i - ii;
            attn_mask[i * batch.n_tokens + ii] = 0.0f;  //ii > i ? -INFINITY : 0.0f;
            pos_bucket[i * batch.n_tokens + ii] =
                (uint32_t) (rpos > 0 ? n_buckets : 0) +
                (ab_rpos < max_exact ?
                     ab_rpos :
                     std::min(
                         (n_buckets - 1),
                         (max_exact + (int) ((log((ab_rpos / max_exact)) / logarithmic_denominator) * max_exact))));
        }
    }
}

void t5_runner::run(uint32_t * input_tokens, uint32_t sequence_length, tts_response * outputs) {
    t5_ubatch batch;
    batch.input_tokens = input_tokens;
    batch.n_tokens     = sequence_length;
    ggml_backend_sched_reset(t5ctx->sched);

    const size_t prev_size = t5ctx->buf_output ? ggml_backend_buffer_get_size(t5ctx->buf_output) : 0;
    const size_t new_size  = model->max_context_length * model->output_size * sizeof(float);

    if (!t5ctx->buf_output || prev_size < new_size) {
        if (t5ctx->buf_output) {
            ggml_backend_buffer_free(t5ctx->buf_output);
            t5ctx->buf_output = nullptr;
            t5ctx->logits     = nullptr;
        }

        t5ctx->buf_output = ggml_backend_buft_alloc_buffer(t5ctx->backend_cpu_buffer, new_size);
    }

    outputs->data = (float *) ggml_backend_buffer_get_base(t5ctx->buf_output);
    ggml_backend_buffer_clear(t5ctx->buf_output, 0);
    ggml_cgraph * gf     = NULL;
    gf                   = build_t5_graph(batch);
    // the output is always the last tensor in the graph
    ggml_tensor * result = gf->nodes[gf->n_nodes - 1];
    ggml_backend_sched_alloc_graph(t5ctx->sched, gf);
    set_inputs(batch);

    ggml_backend_sched_graph_compute_async(t5ctx->sched, gf);

    t5ctx->get_ggml_node_data(result, outputs->data, batch.n_tokens * sizeof(float) * model->output_size);

    // Reset state for the next token before backend sync, to allow the CPU activities in the reset to
    // overlap with device computation.
    ggml_backend_sched_reset(t5ctx->sched);
    outputs->n_outputs   = sequence_length;
    outputs->hidden_size = model->output_size;
    return;
}

int t5_runner::generate(std::string prompt, tts_response * response) {
    std::vector<uint32_t> tokens;
    tokenizer->tokenize(prompt, tokens);
    tokens.push_back(model->eos_token_id);
    run(tokens.data(), (uint32_t) tokens.size(), response);
    return 0;
}

t5_runner * text_encoder_from_file(std::string file_path, int n_threads, unigram_tokenizer * tokenizer, bool cpu_only) {
    t5_encoder *   model      = new t5_encoder;
    ggml_context * weight_ctx = NULL;

    gguf_init_params params = {
        /*.no_alloc   =*/false,
        /*.ctx        =*/&weight_ctx,
    };
    gguf_context * meta_ctx = gguf_init_from_file(file_path.c_str(), params);
    if (!meta_ctx) {
        TTS_ABORT("%s failed for file %s\n", __func__, file_path.c_str());
    }
    if (!tokenizer) {
        tokenizer = unigram_tokenizer_from_gguf(meta_ctx);
    }
    if (!tokenizer->init) {
        tokenizer->initialize_tokenizer();
    }
    model->setup_from_file(meta_ctx, weight_ctx, cpu_only);

    // TODO: change this weight assignment pattern to mirror llama.cpp
    for (ggml_tensor * cur = ggml_get_first_tensor(weight_ctx); cur; cur = ggml_get_next_tensor(weight_ctx, cur)) {
        model->assign_weight(cur->name, cur);
    }

    t5_context * t5ctx  = build_new_t5_context(model, n_threads, cpu_only);
    t5_runner *  runner = new t5_runner(model, t5ctx, tokenizer);
    runner->prepare_post_load();
    gguf_free(meta_ctx);
    ggml_free(weight_ctx);

    return runner;
}
