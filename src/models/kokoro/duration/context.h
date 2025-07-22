#pragma once
#include <cstdint>

#include "tts_model.h"

struct single_pass_tokenizer;
struct kokoro_model;

struct kokoro_ubatch {
    size_t                            n_tokens;      // the number of tokens in our encoded sequence
    uint32_t *                        input_tokens;  // [n_tokens]
    struct kokoro_duration_response * resp = nullptr;
};

struct kokoro_duration_context : runner_context {
    kokoro_duration_context(kokoro_model * model, int n_threads) : runner_context(n_threads), model(model) {};

    ~kokoro_duration_context() override { ggml_backend_buffer_free(buf_len_output); }

    string                voice = "af_alloy";
    kokoro_model *        model;
    ggml_backend_buffer_t buf_len_output = nullptr;

    size_t  logits_size = 0;  // capacity (of floats) for logits
    float * logits      = nullptr;
    float * lens        = nullptr;

    ggml_tensor * inp_tokens;
    ggml_tensor * positions;
    ggml_tensor * attn_mask;
    ggml_tensor * token_types = nullptr;

    void build_schedule() { runner_context::build_schedule(model->max_duration_nodes() * 5); }
};

kokoro_duration_context * build_new_duration_kokoro_context(kokoro_model * model, int n_threads, bool use_cpu = true);

struct kokoro_duration_response {
    size_t  n_outputs;
    float * lengths;
    float * hidden_states;
};

// This struct is intended to manage graph and compute for the duration prediction portion of the kokoro model.
// Duration computation and speech generation are separated into distinct graphs because the precomputed graph structure of ggml doesn't
// support the tensor dependent views that would otherwise be necessary.
struct kokoro_duration_runner : tts_runner {
    explicit kokoro_duration_runner(/* shared */ kokoro_model * model, kokoro_duration_context * context,
                                    single_pass_tokenizer * tokenizer) :
        tokenizer{ tokenizer },
        model{ model },
        kctx{ context } {};

    ~kokoro_duration_runner() {
        if (ctx) {
            ggml_free(ctx);
        }
        delete kctx;
    }
    struct single_pass_tokenizer * tokenizer;
    kokoro_model *                 model;
    kokoro_duration_context *      kctx;

    void init_build() { tts_runner::init_build(&kctx->buf_compute_meta); }

    void          prepare_post_load();
    kokoro_ubatch build_worst_case_batch();
    void          set_inputs(kokoro_ubatch & batch);
    ggml_cgraph * build_kokoro_duration_graph(kokoro_ubatch & batch);
    void          run(kokoro_ubatch & ubatch);
};
