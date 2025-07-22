#pragma once

#include "../../../include/tokenizer.h"
#include "decoder/dac/context.h"
#include "kv.h"
#include "sampler.h"
#include "tts_model.h"

struct parler_tts_model;

struct parler_context : runner_context {
    parler_context(parler_tts_model * model, int n_threads) : runner_context(n_threads), model(model) {};
    parler_tts_model * model;
    vector<bool>              eos_seen;

    bool use_cache = true;

    size_t   output_size      = 0;  /// capacity (of tokens positions) for the output buffers
    int32_t  n_outputs        = 0;  /// number of actually-used outputs in the current ubatch or last logical batch
    uint32_t current_position = 0;  /// current position in the active sequence
    uint32_t prompt_end_position =
        0;  // the position of the text prompt termination (used for adjusting the cache when incrementally generating)
    int32_t seq_id;  /// a unique identifier associated with the active sequence.

    vector<uint32_t> output_tokens;

    size_t  logits_size = 0;  /// capacity (of floats) for logits
    float * logits      = nullptr;

    ggml_tensor * inp_tokens;
    ggml_tensor * audio_inp_tokens;
    ggml_tensor * positions;
    ggml_tensor * attn_mask;
    ggml_tensor * attn_mask_cross;

    void reset(int32_t n_output_heads);
};

struct parler_ubatch {
    bool       audio_generation;  /// whether we are receiving codebook decoded tokens or text tokens
    size_t     n_tokens;          /// total sentence tokens
    size_t     n_audio_tokens;    /// total audio tokens
    /// for just audio tokens the sequence length should be the total_tokens / num_heads;
    /// in general this should be n_tokens + n_audio_tokens / num_heads
    size_t     sequence_length;
    uint32_t * tokens;            // [n_tokens]
    uint32_t * audio_tokens;      // [n_audio_tokens]
    uint32_t * positions;         // [sequence_length]
    uint32_t * true_order;
    int        current_step = 0;  /// total_generations
};

parler_context * build_new_parler_context(parler_tts_model * model, int n_threads, bool use_cpu = true);

// This struct is intended to support end-to-end TTS generation.
// As such, it manages the parler tts model compilation, compute and generation process,
// the tokenization and sampling process, and uses the dac_runner struct to encode audio outputs.
struct parler_tts_runner : tts_runner {
    parler_tts_runner(parler_tts_model * model, dac_runner * dac, parler_context * pctx, unigram_tokenizer * ut,
                      sampler * samp, parler_kv_cache * cache);

    ~parler_tts_runner() override;

    parler_tts_model *  model;
    dac_runner *        dac;
    parler_context *    pctx;
    unigram_tokenizer * tokenizer;
    parler_kv_cache *   kv_self = nullptr;
    sampler *           samp;

    void init_build() { tts_runner::init_build(&pctx->buf_compute_meta); }

    void          configure_generation(const generation_configuration & config);
    void          assign_weight(string name, ggml_tensor * tensor);
    parler_ubatch build_worst_case_batch();
    ggml_cgraph * build_parler_graph(parler_ubatch & batch);
    void          set_inputs(parler_ubatch & batch);
    int           decode(parler_ubatch & batch);
    void          prepare_post_load();
    bool          adjust_for_sequence_continuation(parler_ubatch & batch);
    int           generate(string sentence, tts_response * response, int32_t seq_id = -1);
    bool          check_stopping();
    void          adjust_output_tokens(vector<uint32_t> & output_tokens, vector<uint32_t> & filtered);
    int           generate_from_batch(parler_ubatch & batch, tts_response * output);
    int           generate_audio_tokens(string sentence);
    void update_conditional_prompt(const char * file_path, const char * prompt, int n_threads, bool cpu_only = true);
};
