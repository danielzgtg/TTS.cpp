#include "context.h"
#include "phonemizer.h"

kokoro_runner::~kokoro_runner() {
    if (ctx) {
        ggml_free(ctx);
    }
    delete drunner;
    delete kctx;
    delete phmzr;
}

void kokoro_runner::set_inputs(kokoro_ubatch & batch, uint32_t total_size) {
    random_gen(total_size * model->up_sampling_factor * (model->harmonic_num + 1),
               ((float *) kctx->uv_noise_data->data) + 4);
    ((float *) kctx->uv_noise_data->data)[0] = model->voice_threshold;
    ((float *) kctx->uv_noise_data->data)[1] = model->noise_std;
    ((float *) kctx->uv_noise_data->data)[2] = model->sin_amp;
    ((float *) kctx->uv_noise_data->data)[3] = model->sin_amp / 3.0f;
    compute_window_squared_sum(model->true_n_fft, model->stft_hop,
                               total_size * model->up_sampling_factor / model->stft_hop,
                               (float *) kctx->window_sq_sum->data, (float *) model->decoder->generator->window->data);
    kctx->sequence_length = batch.n_tokens;
    kctx->total_duration  = total_size;
    ggml_backend_tensor_set(kctx->inp_tokens, batch.input_tokens, 0,
                            batch.n_tokens * ggml_element_size(kctx->inp_tokens));
    ggml_backend_tensor_set(kctx->duration_pred, batch.resp->hidden_states, 0,
                            batch.n_tokens * (model->duration_hidden_size + model->style_half_size) *
                                ggml_element_size(kctx->duration_pred));
    float * d       = nullptr;
    float   running = 0;
    d               = (float *) kctx->duration_mask->data;
    for (uint32_t i = 0; i < batch.n_tokens; i++) {
        float next_running = running + batch.resp->lengths[i];
        for (uint32_t ii = 0; ii < total_size; ii++) {
            d[i * total_size + ii] = ii >= running && ii < next_running ? 1.0f : 0.0f;
        }
        running = next_running;
    }
}

void kokoro_runner::run(kokoro_ubatch & batch, tts_response * outputs) {
    batch.resp = new kokoro_duration_response;
    drunner->run(batch);

    ggml_backend_sched_reset(kctx->sched);

    const size_t prev_size    = kctx->buf_output ? ggml_backend_buffer_get_size(kctx->buf_output) : 0;
    uint32_t     total_length = 0;
    for (int i = 0; i < batch.resp->n_outputs; i++) {
        total_length += (uint32_t) batch.resp->lengths[i];
    }
    const size_t new_size = total_length * model->up_sampling_factor * sizeof(float);

    if (!kctx->buf_output || prev_size < new_size) {
        if (kctx->buf_output) {
            ggml_backend_buffer_free(kctx->buf_output);
            kctx->buf_output = nullptr;
            kctx->logits     = nullptr;
        }
        kctx->buf_output = ggml_backend_buft_alloc_buffer(kctx->backend_cpu_buffer, new_size);
    }

    outputs->data = (float *) ggml_backend_buffer_get_base(kctx->buf_output);
    ggml_backend_buffer_clear(kctx->buf_output, 0);

    kctx->sequence_length = batch.n_tokens;
    kctx->total_duration  = total_length;

    ggml_cgraph * gf = NULL;
    gf               = build_kokoro_graph(batch);

    // the output is always the last tensor in the graph
    ggml_tensor * output = gf->nodes[gf->n_nodes - 1];

    ggml_backend_sched_alloc_graph(kctx->sched, gf);

    set_inputs(batch, total_length);

    ggml_backend_sched_graph_compute_async(kctx->sched, gf);

    kctx->get_ggml_node_data(output, outputs->data, new_size);

    // Reset state for the next token before backend sync, to allow the CPU activities in the reset to
    // overlap with device computation.
    ggml_backend_sched_reset(kctx->sched);
    outputs->n_outputs = total_length * model->up_sampling_factor;
    free(batch.resp);
    return;
}

void kokoro_runner::assign_weight(std::string name, ggml_tensor * tensor) {
    model->assign_weight(name, tensor);
}

/*
 * #tokenize_chunks is used to split up a larger than max context size (512) token prompt into discrete
 * blocks for generation. This solution, in accordance with Kokoro's pyTorch implementation, splits
 * the prompt by sentence when possible (this can result in slower inference but generally produces cleaner
 * speech). If a disinct sentence is too long, then it splits at the nearest space.
 */
std::vector<std::vector<uint32_t>> kokoro_runner::tokenize_chunks(std::vector<std::string> clauses) {
    std::vector<std::vector<uint32_t>> chunks;
    for (auto clause : clauses) {
        clause = strip(clause);
        if (clause.empty()) {
            continue;
        }
        std::vector<uint32_t> tokens;
        tokens.push_back(model->bos_token_id);
        tokenizer->tokenize(clause, tokens);
        // if there are more clause tokens than the max context length then try to split by space tokens.
        // To be protective, split mid-word when there are no spaces (this should never happen).
        if (tokens.size() > model->max_context_length - 2) {
            // we skip the first token here becuase it is the bos token.
            int last_space_token = 1;
            int last_split       = 1;
            for (int i = 1; i < tokens.size(); i++) {
                if (tokens[i] == model->space_token_id) {
                    last_space_token = i;
                }
                if ((i - last_split) + chunks.back().size() >= model->max_context_length - 1) {
                    if (last_space_token > last_split) {
                        std::vector<uint32_t> portion = { model->bos_token_id };
                        portion.insert(portion.end(), tokens.begin() + last_split, tokens.begin() + last_space_token);
                        portion.push_back(model->eos_token_id);
                        chunks.push_back(portion);
                        last_split = last_space_token;
                    } else {
                        std::vector<uint32_t> portion = { model->bos_token_id };
                        portion.insert(portion.end(), tokens.begin() + last_split, tokens.begin() + i + 1);
                        portion.push_back(model->eos_token_id);
                        chunks.push_back(portion);
                        last_split = i + 1;
                    }
                }
            }
            if (last_split + 1 < tokens.size()) {
                std::vector<uint32_t> portion = { model->bos_token_id };
                portion.insert(portion.end(), tokens.begin() + last_split, tokens.end());
                portion.push_back(model->eos_token_id);
                chunks.push_back(portion);
            }
        } else {
            tokens.push_back(model->eos_token_id);
            chunks.push_back(tokens);
        }
    }
    return chunks;
}

int kokoro_runner::generate(std::string prompt, tts_response * response, std::string voice, std::string voice_code) {
    if (model->voices.find(voice) == model->voices.end()) {
        TTS_ABORT("Failed to find Kokoro voice '%s' aborting.\n", voice.c_str());
    } else {
        // if the language changed then we should change the phonemization voice
        if (phmzr->mode == ESPEAK && kctx->voice[0] != voice[0]) {
            if (voice_code.empty()) {
                voice_code = get_espeak_id_from_kokoro_voice(voice.c_str());
            }
            update_voice(voice_code);
        }
        kctx->voice          = voice;
        drunner->kctx->voice = voice;
    }
    // replace all non-sentence terminating characters with '--' which espeak will treat as a pause.
    // We preserve the other punctuation for cleaner chunking pre-tokenization
    prompt                        = replace_any(prompt, ",;:", "--");
    prompt                        = replace_any(prompt, "\n", " ");
    std::string phonemized_prompt = phmzr->text_to_phonemes(prompt);

    // Kokoro users a utf-8 single character tokenizer so if the size of the prompt is smaller than the max context length without the
    // beginning of sentence and end of sentence tokens then we can compute it all at once.
    if (phonemized_prompt.size() < model->max_context_length - 2) {
        // we preserved punctuation and Kokoro interprets these tokens as end of sentence tokens, so we have to remove them for all-at-once compute.
        phonemized_prompt = strip(replace_any(phonemized_prompt, ".!?", ""));
        if (phonemized_prompt.empty()) {
            return 0;
        }
        std::vector<uint32_t> tokens;
        tokens.push_back(model->bos_token_id);
        tokenizer->tokenize(phonemized_prompt, tokens);
        tokens.push_back(model->eos_token_id);
        kokoro_ubatch batch;
        batch.n_tokens     = tokens.size();
        batch.input_tokens = tokens.data();
        run(batch, response);
    } else {
        // TODO: determine the performance to memory trade off in using a batched compute approach verse this chunking approach.
        // This approach is likely to be slower than a batched approach, but given the already huge memory overhead of Kokoro's graph it
        // might be preferable to use this chunking approach.
        std::vector<std::string> clauses = split(phonemized_prompt, ".!?");
        for (auto tokens : tokenize_chunks(clauses)) {
            kokoro_ubatch batch;
            batch.n_tokens         = tokens.size();
            batch.input_tokens     = tokens.data();
            tts_response * partial = new tts_response;
            run(batch, partial);
            append_to_response(response, partial);
        }
    }
    return 0;
}

kokoro_duration_context * build_new_duration_kokoro_context(kokoro_model * model, int n_threads, bool use_cpu) {
    kokoro_duration_context * kctx = new kokoro_duration_context(model, n_threads);
    if (!use_cpu) {
#ifdef GGML_USE_METAL
        kctx->backend = ggml_backend_metal_init();
#endif
    }
    kctx->backend_cpu = ggml_backend_cpu_init();
    kctx->set_threads();
    kctx->build_schedule();
    kctx->buf_compute_meta.resize(ggml_tensor_overhead() * model->max_duration_nodes() * 5 +
                                  ggml_graph_overhead_custom(model->max_duration_nodes() * 5, false));
    return kctx;
}

size_t kokoro_model::max_gen_nodes() {
    return std::max<size_t>(8192, generation_node_counter * 2);
}

size_t kokoro_model::max_duration_nodes() {
    return std::max<size_t>(8192, duration_node_counter * 2);
}

kokoro_context * build_new_kokoro_context(kokoro_model * model, int n_threads, bool use_cpu) {
    kokoro_context * kctx = new kokoro_context(model, n_threads);
    if (!use_cpu) {
#ifdef GGML_USE_METAL
        kctx->backend = ggml_backend_metal_init();
#endif
    }
    kctx->backend_cpu = ggml_backend_cpu_init();
    kctx->set_threads();
    kctx->build_schedule();
    kctx->buf_compute_meta.resize(ggml_tensor_overhead() * model->max_gen_nodes() * 30 +
                                  ggml_graph_overhead_custom(model->max_gen_nodes() * 30, false));
    return kctx;
}
