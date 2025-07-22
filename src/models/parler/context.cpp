#include "context.h"

parler_context * build_new_parler_context(parler_tts_model * model, int n_threads, bool use_cpu) {
    parler_context * pctx = new parler_context(model, n_threads);
    if (!use_cpu) {
#ifdef GGML_USE_METAL
        pctx->backend = ggml_backend_metal_init();
#endif
    }
    pctx->eos_seen.reserve(model->n_output_heads);
    pctx->backend_cpu = ggml_backend_cpu_init();
    pctx->set_threads();
    pctx->build_schedule(model->max_nodes());
    pctx->buf_compute_meta.resize(ggml_tensor_overhead() * model->max_nodes() +
                                  ggml_graph_overhead_custom(model->max_nodes(), false));
    return pctx;
}

parler_tts_runner::parler_tts_runner(parler_tts_model * model, dac_runner * dac, parler_context * pctx,
                                     unigram_tokenizer * ut, sampler * samp, parler_kv_cache * cache) :
    model(model),
    dac(dac),
    pctx(pctx),
    tokenizer(ut),
    samp(samp),
    kv_self(cache) {}

parler_tts_runner::~parler_tts_runner() {
    ggml_free(ctx);
    model->free();
    delete model;
    delete kv_self;
    delete dac_runner;
    delete pctx;
    delete sampler;
}

void parler_context::reset(int32_t n_output_heads) {
    n_outputs           = 0;
    prompt_end_position = 0;
    current_position    = 0;
    output_size         = 0;
    output_tokens.clear();
    eos_seen.clear();
    for (int i = 0; i < (int) n_output_heads; i++) {
        eos_seen.push_back(false);
    }
}

bool parler_tts_runner::adjust_for_sequence_continuation(parler_ubatch & batch) {
    return false;  // not implemneted
}

int parler_tts_runner::generate(std::string sentence, tts_response * output, int32_t seq_id) {
    parler_ubatch batch = batch_from_sentence(sentence, model, tokenizer);
    pctx->reset(model->n_output_heads);
    sampler->reset();
    if (pctx->seq_id != seq_id || seq_id == -1) {
        seq_id                 = std::mt19937(std::random_device{}())();
        pctx->current_position = 0;
        if (!kv_self) {
            kv_self = new parler_kv_cache;
            if (!parler_kv_cache_init(kv_self, model, pctx, seq_id)) {
                return 1;
            }
        }
    } else {
        if (!adjust_for_sequence_continuation(batch)) {
            return 2;
        }
    }
    return generate_from_batch(batch, output);
}

void parler_tts_runner::update_conditional_prompt(str file_path, str prompt, int n_threads, bool cpu_only) {
    t5_runner *    text_encoder = text_encoder_from_file(file_path, n_threads, tokenizer, cpu_only);
    tts_response * response;
    text_encoder->generate(prompt, response);
    model->prep_cross_key_values(n_threads, response);
    delete text_encoder;
}

void parler_tts_runner::configure_generation(const generation_configuration & config) {
    sampler->temperature        = config.temperature;
    sampler->repetition_penalty = config.repetition_penalty;
    sampler->do_sample          = config.sample;
    sampler->top_k              = config.top_k;
    sampler->top_p              = config.top_p;
    model->use_cross_attn       = config.use_cross_attn;
}

void parler_tts_runner::set_inputs(parler_ubatch & batch) {
    if (batch.audio_generation) {
        ggml_backend_tensor_set(pctx->audio_inp_tokens, batch.audio_tokens, 0,
                                batch.n_audio_tokens * ggml_element_size(pctx->audio_inp_tokens));
    } else {
        ggml_backend_tensor_set(pctx->inp_tokens, batch.tokens, 0,
                                batch.n_tokens * ggml_element_size(pctx->inp_tokens));
    }
    ggml_backend_tensor_set(pctx->positions, batch.positions, 0,
                            batch.sequence_length * ggml_element_size(pctx->positions));
    float * d        = nullptr;
    d                = (float *) pctx->attn_mask->data;
    uint32_t max_pos = pctx->current_position + batch.sequence_length;
    for (int i = 0; i < batch.sequence_length; i++) {
        uint32_t pos = batch.positions[i];
        for (int ii = 0; ii < max_pos; ii++) {
            d[i * max_pos + ii] = ii > pos ? -INFINITY : 0.0f;
        }
    }

    if (model->use_cross_attn) {
        float * d2 = nullptr;
        d2         = (float *) pctx->attn_mask_cross->data;
        for (int i = 0; i < model->n_encode_length; i++) {
            for (int ii = 0; ii < batch.sequence_length; ii++) {
                d2[i * batch.sequence_length + ii] = 0.0f;
            }
        }
    }
}

int parler_tts_runner::decode(parler_ubatch & batch) {
    ggml_backend_sched_reset(pctx->sched);

    pctx->output_tokens.reserve(model->max_generation_size);

    const size_t logits_size = model->output_vocab_size * model->max_generation_size * model->n_output_heads;
    const size_t prev_size   = pctx->buf_output ? ggml_backend_buffer_get_size(pctx->buf_output) : 0;
    const size_t new_size    = logits_size * sizeof(float);

    if (!pctx->buf_output || prev_size < new_size) {
        if (pctx->buf_output) {
            ggml_backend_buffer_free(pctx->buf_output);
            pctx->buf_output = nullptr;
            pctx->logits     = nullptr;
        }

        pctx->buf_output = ggml_backend_buft_alloc_buffer(pctx->backend_cpu_buffer, new_size);
    }

    pctx->logits = (float *) ggml_backend_buffer_get_base(pctx->buf_output);
    //ggml_backend_buffer_clear(pctx->buf_output, 0);

    ggml_cgraph * gf = build_parler_graph(batch);

    // the output is always the last tensor in the graph
    ggml_tensor * res = gf->nodes[gf->n_nodes - 1];
    ggml_backend_sched_alloc_graph(pctx->sched, gf);

    // use the sequence_length variable here so that audio input tokens are handled correctly.
    size_t n_outputs_new = batch.sequence_length;

    set_inputs(batch);
    ggml_backend_sched_graph_compute_async(pctx->sched, gf);

    float * logits_out = pctx->logits + pctx->n_outputs * model->output_vocab_size * model->n_output_heads;
    pctx->get_ggml_node_data(res, logits_out,
                             n_outputs_new * model->output_vocab_size * model->n_output_heads * sizeof(float));

    // set to total number of outputs in the batch*/
    pctx->n_outputs += n_outputs_new;

    // Reset state for the next token before backend sync, to allow the CPU activities in the reset to
    // overlap with device computation.
    ggml_backend_sched_reset(pctx->sched);

    return 0;
}

bool parler_tts_runner::check_stopping() {
    int32_t token_position = (int32_t) pctx->output_tokens.size() - (int32_t) model->n_output_heads;
    if (token_position < 0) {
        return false;
    }
    if (pctx->current_position >= model->max_generation_size) {
        return true;
    }

    bool channels_complete = true;
    for (int i = 0; i < model->n_output_heads; i++) {
        pctx->eos_seen[i] = pctx->eos_seen[i] || pctx->output_tokens[token_position + i] == model->eos_token_id;
        if (channels_complete) {
            channels_complete = pctx->eos_seen[i];
        }
    }
    return channels_complete;
}

void parler_tts_runner::adjust_output_tokens(std::vector<uint32_t> & output_tokens, std::vector<uint32_t> & filtered) {
    // currently this is applying sliding window over the heads and filtering out bad tokens.
    // If we convert the DAC model's quantizer layers to support by row + column embeddings then we will need to transpose
    // the heads and the sequence here, but right now simplying using a strided view is more peformant.
    size_t size = output_tokens.size();
    filtered.reserve(size);
    for (int i = 0; i < size / model->n_output_heads; i++) {
        bool remove = false;
        for (int ii = 0; ii < model->n_output_heads; ii++) {
            int next_index = i * model->n_output_heads + ii * model->n_output_heads + ii;
            if (next_index > size || output_tokens[next_index] >= model->audio_vocab_size) {
                remove = true;
                break;
            }
        }
        if (!remove) {
            for (int ii = 0; ii < model->n_output_heads; ii++) {
                int next_index = i * model->n_output_heads + ii * model->n_output_heads + ii;
                if (next_index > size) {
                    filtered.push_back(model->eos_token_id);
                } else {
                    filtered.push_back(output_tokens[next_index]);
                }
            }
        }
    }
}

int parler_tts_runner::generate_from_batch(parler_ubatch & batch, tts_response * output) {
    std::vector<uint32_t> next_decoder_token_ids;
    next_decoder_token_ids.reserve(model->n_output_heads);

    while (!check_stopping()) {
        int state = decode(batch);
        if (state != 0) {
            return state;
        }
        if (!batch.audio_generation) {
            pctx->prompt_end_position += batch.sequence_length;
        }
        if (batch.audio_generation) {
            sampler->sample(pctx->logits + pctx->current_position * model->n_output_heads * model->output_vocab_size,
                            pctx->output_tokens);
        }
        pctx->current_position += batch.sequence_length;
        next_decoder_token_ids.clear();
        uint32_t * last_outputs =
            (pctx->output_tokens.data() + (int) pctx->output_tokens.size() - model->n_output_heads);
        for (int i = 0; i < model->n_output_heads; i++) {
            next_decoder_token_ids.push_back(batch.current_step > i ?
                                                 pctx->eos_seen[i] ? model->eos_token_id : last_outputs[i] :
                                                 model->bos_token_id);
        }
        batch = parler_ubatch{ true,
                               0,
                               9,
                               1,
                               nullptr,
                               next_decoder_token_ids.data(),
                               &pctx->current_position,
                               nullptr,
                               batch.current_step + 1 };
    }

    std::vector<uint32_t> filtered_output_tokens;
    adjust_output_tokens(pctx->output_tokens, filtered_output_tokens);
    dac_runner->run(filtered_output_tokens.data(), (int32_t) filtered_output_tokens.size() / model->n_output_heads,
                    output);
    return 0;
}
