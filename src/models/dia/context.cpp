#include "dia_model.h"

void dia_context::reset() {
    current_position = 0;
    prompt_size      = 0;
    output_tokens.clear();
    delay_steps = -1;
}

dia_context * build_new_dia_context(dia_model * model, int n_threads, bool use_cpu) {
    dia_context * dctx = new dia_context(model, n_threads);
    if (!use_cpu) {
#ifdef GGML_USE_METAL
        dctx->backend = ggml_backend_metal_init();
#endif
    }
    dctx->backend_cpu = ggml_backend_cpu_init();
    dctx->set_threads();
    dctx->build_schedule();
    dctx->buf_compute_meta.resize(ggml_tensor_overhead() * model->max_nodes() +
                                  ggml_graph_overhead_custom(model->max_nodes(), false));
    return dctx;
}


void dia_runner::configure_generation(const generation_configuration & config) {
    GGML_ASSERT(config.max_tokens == 0 || config.max_tokens > model->max_delay);
    decode_sampler->temperature        = config.temperature;
    decode_sampler->repetition_penalty = config.repetition_penalty;
    decode_sampler->do_sample          = config.sample;
    decode_sampler->top_k              = config.top_k;
    decode_sampler->top_p              = config.top_p;
    dctx->max_generation_size = config.max_tokens > model->max_delay ? config.max_tokens : model->max_generation_size;
}

void dia_runner::set_inputs(dia_ubatch & batch) {
    if (batch.encoder_step) {
        ggml_backend_tensor_set(dctx->inp_tokens, batch.tokens.data(), 0,
                                batch.tokens.size() * ggml_element_size(dctx->inp_tokens));
        int32_t * ep   = (int32_t *) dctx->encode_positions->data;
        float *   mask = (float *) dctx->encode_attn_mask->data;
        for (int i = 0; i < model->max_encoder_context_length; i++) {
            ep[i] = i;
            for (int ii = 0; ii < model->max_encoder_context_length; ii++) {
                if (i < batch.sentence_length) {
                    mask[i * model->max_encoder_context_length + ii] = ii < batch.sentence_length ? 0.0 : -INFINITY;
                } else {
                    mask[i * model->max_encoder_context_length + ii] = ii >= batch.sentence_length ? 0.0 : -INFINITY;
                }
            }
        }
    }
    // The audio tokens need to be repeated in the input in order to support cfg-scaling. I.E we need duplicate inputs for conditional and unconditional logits.
    ggml_backend_tensor_set(dctx->audio_inp_tokens, batch.audio_tokens.data(), 0,
                            batch.audio_tokens.size() * ggml_element_size(dctx->audio_inp_tokens));
    ggml_backend_tensor_set(dctx->audio_inp_tokens, batch.audio_tokens.data(),
                            batch.audio_tokens.size() * ggml_element_size(dctx->audio_inp_tokens),
                            batch.audio_tokens.size() * ggml_element_size(dctx->audio_inp_tokens));
    ((int32_t *) dctx->positions->data)[0] = dctx->current_position;
}

int dia_runner::decode(dia_ubatch & batch) {
    if (batch.encoder_step) {
        dctx->prompt_size = batch.sentence_length;
        dctx->output_tokens.reserve(dctx->max_generation_size * model->n_output_heads);
    }
    ggml_backend_sched_reset(dctx->sched);

    const size_t logits_size = model->output_vocab_size * dctx->max_generation_size * model->n_output_heads;
    const size_t prev_size   = dctx->buf_output ? ggml_backend_buffer_get_size(dctx->buf_output) : 0;
    const size_t new_size    = logits_size * sizeof(float);

    if (!dctx->buf_output || prev_size < new_size) {
        if (dctx->buf_output) {
            ggml_backend_buffer_free(dctx->buf_output);
            dctx->buf_output = nullptr;
            dctx->logits     = nullptr;
        }

        dctx->buf_output = ggml_backend_buft_alloc_buffer(dctx->backend_cpu_buffer, new_size);
    }

    dctx->logits = (float *) ggml_backend_buffer_get_base(dctx->buf_output);

    ggml_cgraph * gf = build_dia_graph(batch);

    // the output is always the last tensor in the graph
    ggml_tensor * res     = gf->nodes[gf->n_nodes - 1];
    string        resname = ggml_get_name(res);
    ggml_backend_sched_alloc_graph(dctx->sched, gf);

    set_inputs(batch);

    ggml_backend_sched_graph_compute_async(dctx->sched, gf);

    float * logits_out = dctx->logits + dctx->current_position * model->output_vocab_size * model->n_output_heads;
    dctx->get_ggml_node_data(res, logits_out, model->output_vocab_size * model->n_output_heads * sizeof(float));

    // Reset state for the next token before backend sync, to allow the CPU activities in the reset to
    // overlap with device computation.
    ggml_backend_sched_reset(dctx->sched);

    return 0;
}

bool dia_runner::check_stopping(dia_ubatch & batch) {
    if (dctx->delay_steps == -1 && (batch.audio_tokens[0] == model->eos_token_id ||
                                    dctx->current_position >= dctx->max_generation_size - model->max_delay)) {
        dctx->delay_steps = model->max_delay;
    }

    if (dctx->delay_steps > 0) {
        int step_after_eos = model->max_delay - dctx->delay_steps;
        for (int i = 0; i < model->delay_pattern.size(); i++) {
            if (step_after_eos == model->delay_pattern[i]) {
                batch.audio_tokens[i] = model->eos_token_id;
            } else if (step_after_eos > model->delay_pattern[i]) {
                batch.audio_tokens[i] = model->pad_token_id;
            }
        }
        dctx->delay_steps -= 1;
    }
    return dctx->delay_steps == 0;
}

void dia_runner::adjust_output_tokens(vector<uint32_t> & output_tokens, vector<uint32_t> & filtered) {
    // currently this is applying sliding window over the heads and filtering out bad tokens.
    // If we convert the DAC model's quantizer layers to support by row + column embeddings
    // then we will need to transpose the heads and the sequence here,
    // but right now simplifying using a strided view is more performant.
    size_t size = output_tokens.size();
    filtered.reserve(size);
    for (int i = 0; i < (size / model->n_output_heads) - model->max_delay; i++) {
        bool skip_step = false;
        for (int ii = 0; ii < model->n_output_heads; ii++) {
            int next_index = i * model->n_output_heads + model->delay_pattern[ii] * model->n_output_heads + ii;
            if (next_index > size || output_tokens[next_index] >= model->audio_vocab_size) {
                skip_step = true;
                break;
            }
        }
        if (!skip_step) {
            for (int ii = 0; ii < model->n_output_heads; ii++) {
                int next_index = i * model->n_output_heads + model->delay_pattern[ii] * model->n_output_heads + ii;
                filtered.push_back(output_tokens[next_index]);
            }
        }
    }
}

int dia_runner::generate_from_batch(dia_ubatch & batch, tts_response * output) {
    while (!check_stopping(batch)) {
        if (const int state = decode(batch); state != 0) {
            return state;
        }
        decode_sampler->sample(dctx->logits + dctx->current_position * model->n_output_heads * model->output_vocab_size,
                               dctx->output_tokens);
        dctx->current_position += batch.sequence_length;
        batch = dia_ubatch{ 1 };
        uint32_t * last_outputs =
            (dctx->output_tokens.data() + (int) dctx->output_tokens.size() - model->n_output_heads);
        batch.audio_tokens.reserve(model->n_output_heads);
        for (int i = 0; i < model->n_output_heads; i++) {
            batch.audio_tokens.push_back(dctx->current_position > i ? last_outputs[i] : model->bos_token_id);
        }
    }

    vector<uint32_t> filtered_output_tokens;
    adjust_output_tokens(dctx->output_tokens, filtered_output_tokens);

    dac_runner->run(filtered_output_tokens.data(), (int32_t) filtered_output_tokens.size() / model->n_output_heads,
                    output);
    return 0;
}

int dia_runner::generate(string sentence, tts_response * output) {
    dia_ubatch batch = batch_from_sentence(sentence);
    dctx->reset();
    decode_sampler->reset();
    dctx->current_position = 0;
    if (!kv_cross_self) {
        kv_cross_self = new dia_kv_cache;
        if (!dia_kv_cache_init(kv_cross_self, model, dctx)) {
            return 1;
        }
    }
    return generate_from_batch(batch, output);
}
