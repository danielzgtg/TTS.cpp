#pragma once

struct t5_context : runner_context {
    t5_context(t5_encoder * model, int n_threads) : runner_context(n_threads), model(model) {};

    t5_encoder * model;

    size_t  logits_size = 0;  // capacity (of floats) for logits
    float * logits      = nullptr;

    ggml_tensor * inp_tokens;
    ggml_tensor * positions;
    ggml_tensor * attn_mask;
    ggml_tensor * inp_pos_bucket;

    void build_schedule() { runner_context::build_schedule(model->max_nodes()); }
};

t5_context * build_new_t5_context(t5_encoder * model, int n_threads, bool use_cpu = true);

struct t5_ubatch {
    size_t     n_tokens;      // the number of tokens in our encoded sequence
    uint32_t * input_tokens;  // [n_tokens]
};

// This struct is intended to manage the t5 encoder model's graph compilation and compute function.
struct t5_runner : tts_runner {
    t5_runner(t5_encoder * model, t5_context * context, unigram_tokenizer * tokenizer) :
        model(model),
        t5ctx(context),
        tokenizer(tokenizer) {};

    ~t5_runner() {
        ggml_free(ctx);
        model->free();
        delete model;
        delete t5ctx;
    }

    unigram_tokenizer * tokenizer;
    t5_encoder *        model;
    t5_context *        t5ctx;

    void init_build() { tts_runner::init_build(&t5ctx->buf_compute_meta); }

    void          prepare_post_load() override;
    t5_ubatch     build_worst_case_batch();
    void          set_inputs(t5_ubatch & batch);
    ggml_cgraph * build_t5_graph(t5_ubatch & batch);
    void          run(uint32_t * input_tokens, uint32_t sequence_length, tts_response * outputs);
    int           generate(std::string prompt, tts_response * response);
};

t5_runner * text_encoder_from_file(std::string file_path, int n_threads, unigram_tokenizer * tokenizer,
                                   bool cpu_only = true);
