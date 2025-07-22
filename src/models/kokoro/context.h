#pragma once

#include "tts_model.h"

struct kokoro_duration_runner;
struct kokoro_model;
struct kokoro_ubatch;
struct phonemizer;
struct single_pass_tokenizer;

struct kokoro_context : runner_context {
    explicit kokoro_context(kokoro_model * model, int n_threads) : runner_context{ n_threads }, model{ model } {}

    std::string voice = "af_alloy";

    kokoro_model * model;

    uint32_t total_duration;
    uint32_t sequence_length;

    size_t  logits_size = 0;  // capacity (of floats) for logits
    float * logits      = nullptr;

    ggml_tensor * inp_tokens;
    ggml_tensor * duration_pred;
    ggml_tensor * duration_mask;
    ggml_tensor * window_sq_sum;  // needs to be calculatd from the generator window.
    ggml_tensor * uv_noise_data;

    void build_schedule() { runner_context::build_schedule(model->max_gen_nodes() * 30); }
};

kokoro_context * build_new_kokoro_context(kokoro_model * model, int n_threads, bool use_cpu = true);

// This manages the graph compilation of computation for the Kokoro model.
struct kokoro_runner : tts_runner {
    explicit kokoro_runner(unique_ptr<kokoro_model> && model, kokoro_context * context,
                           single_pass_tokenizer * tokenizer, kokoro_duration_runner * drunner, phonemizer * phmzr) :
        tokenizer{ tokenizer },
        model{ move(model) },
        kctx{ context },
        drunner{ drunner },
        phmzr{ phmzr } {
        sampling_rate = 24000.0f;
    };

    ~kokoro_runner() override;

    single_pass_tokenizer *  tokenizer;
    unique_ptr<kokoro_model> model;
    kokoro_context *         kctx;
    kokoro_duration_runner * drunner;
    phonemizer *             phmzr;

    string default_voice = "af_alloy";

    void init_build() { tts_runner::init_build(&kctx->buf_compute_meta); }

    vector<vector<uint32_t>> tokenize_chunks(vector<string> clauses);
    void                     assign_weight(string name, ggml_tensor * tensor);
    void                     prepare_post_load();
    kokoro_ubatch            build_worst_case_batch();
    void                     set_inputs(kokoro_ubatch & batch, uint32_t total_size);
    ggml_cgraph *            build_kokoro_graph(kokoro_ubatch & batch);
    void                     run(kokoro_ubatch & batch, tts_response * outputs);
    int                      generate(string prompt, tts_response * response, string voice, string voice_code = "");
};
