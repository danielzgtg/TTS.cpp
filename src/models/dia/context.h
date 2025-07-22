#pragma once
#include "common.h"

struct dia_model;
struct dac_runner;
struct ggml_tensor;

struct dia_context : runner_context {
    dia_context(dia_model * model, int n_threads): runner_context(n_threads), model(model) {
        max_generation_size = model->max_generation_size;
    }

    uint32_t current_position = 0;  // current position in the active sequence
    int delay_steps           = -1; // the max remaining steps to take before terminating; is set after an eos token is seen on the first output channel
    size_t prompt_size        = 0;
    float * logits            = nullptr;

    uint32_t max_generation_size; // this is set by the generation context or defaults to the config set on dia model.

    vector<uint32_t> output_tokens;
    dia_model * model;

    ggml_tensor * inp_tokens;
    ggml_tensor * audio_inp_tokens;
    ggml_tensor * positions;
    ggml_tensor * encode_positions;
    ggml_tensor * encode_attn_mask;
    ggml_tensor * cross_attn_mask;

    void build_schedule() {
        runner_context::build_schedule(model->max_nodes());
    }
    void reset();
};

struct dia_ubatch {
    dia_ubatch(size_t sequence_length, bool encoder_step = false): sequence_length(sequence_length), encoder_step(encoder_step) {};
    bool encoder_step; // whether we are performing the prompt encoding in this step.
    size_t sequence_length; // for just audio tokens the sequence length should be the total_tokens / num_heads; for normal generation this should always be 1.
    size_t sentence_length; // the number of non padded tokens in the conditional context
    vector<uint32_t> tokens; // character tokens for the encoder
    vector<uint32_t> audio_tokens; // audio tokens from the last generation
};

dia_context * build_new_dia_context(dia_model * model, int n_threads, bool use_cpu = true);

// This struct is intended to support end-to-end TTS generation for the Dia model. As such, it manages Dia's model compilation, compute, generation,
// tokenizationm and sampling process, and uses the dac_runner struct to encode audio outputs.
struct dia_runner : public tts_runner {
    dia_runner(dia_model * model, dac_runner * audio_decoder, dia_context * dctx, sampler * samp, dia_kv_cache * cache): model(model), dac_runner(audio_decoder), dctx(dctx), decode_sampler(samp), kv_cross_self(cache) {
        decode_sampler->vocab_size = model->output_vocab_size;
    };
    ~dia_runner() {
        ggml_free(ctx);
        model->free();
        delete model;
        delete kv_cross_self;
        delete dac_runner;
        delete dctx;
        delete decode_sampler;
    }
    dia_model * model;
    dac_runner * dac_runner;
    dia_context * dctx;
    dia_kv_cache * kv_cross_self = nullptr;
    sampler * decode_sampler;

    void init_build() {
        tts_runner::init_build(&dctx->buf_compute_meta);
    }

    void tokenize_sentence(std::string sentence, dia_ubatch & tokens);
    dia_ubatch batch_from_sentence(std::string sentence);
    void configure_generation(const generation_configuration & config);
    void assign_weight(std::string name, ggml_tensor * tensor);
    dia_ubatch build_worst_case_batch();
    ggml_cgraph * build_dia_graph(dia_ubatch & batch);
    void set_inputs(dia_ubatch & batch);
    int decode(dia_ubatch & batch);
    void prepare_post_load();
    int generate(std::string sentence, struct tts_response * response);
    bool check_stopping(dia_ubatch & batch);
    void adjust_output_tokens(std::vector<uint32_t> & output_tokens, std::vector<uint32_t> & filtered);
    int generate_from_batch(dia_ubatch & batch, struct tts_response * output);
};
