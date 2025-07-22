#pragma once

#include "tts_model.h"

struct dac_model;

// the context used for running the dac model
struct dac_context : runner_context {
    dac_context(dac_model * model, int n_threads) : runner_context(n_threads), model(model) {};

    dac_model * model;

    size_t  logits_size = 0;  // capacity (of floats) for logits
    float * logits      = nullptr;

    ggml_tensor * inp_tokens;
};

struct dac_ubatch {
    uint32_t * input_tokens;
    uint32_t   sequence_length;
};

dac_context * build_new_dac_context(dac_model * model, int n_threads, bool use_cpu);

// This struct is intended to manage the dac model's graph compilation and compute function.
struct dac_runner : tts_runner {
    dac_runner(dac_model * model, dac_context * context);

    ~dac_runner() override;

    dac_model *   model;
    dac_context * dctx;

    void init_build() { tts_runner::init_build(&dctx->buf_compute_meta); }

    void          prepare_post_load();
    ggml_cgraph * build_dac_graph(dac_ubatch & batch);
    void          run(uint32_t * input_tokens, uint32_t sequence_length, tts_response * outputs);
};
