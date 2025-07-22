#pragma once

#include "../../../include/tokenizer.h"
#include "ada/model.h"
#include "duration/model.h"
#include "tts_model.h"

// Rather than using ISO 639-2 language codes, Kokoro voice pack specify their corresponding language via their first letter.
// Below is a map that describes the relationship between those designations and espeak-ng's voice identifiers so that the
// appropriate phonemization protocol can inferred from the Kokoro voice.
constexpr auto KOKORO_LANG_TO_ESPEAK_ID{ [] {
    std::array<str, 256> result{};
    result['a'] = "gmw/en-US";
    result['b'] = "gmw/en";
    result['e'] = "roa/es";
    result['f'] = "roa/fr";
    result['h'] = "inc/hi";
    result['i'] = "roa/it";
    result['j'] = "jpx/ja";
    result['p'] = "roa/pt-BR";
    result['z'] = "sit/cmn";
    return result;
}() };

constexpr str get_espeak_id_from_kokoro_voice(str voice) {
    const auto value = KOKORO_LANG_TO_ESPEAK_ID[static_cast<unsigned char>(voice[0])];
    return value ? value : "gmw/en-US";
}

struct kokoro_text_encoder_conv_layer {
    ggml_tensor * norm_gamma;
    ggml_tensor * norm_beta;
    ggml_tensor * conv_weight;
    ggml_tensor * conv_bias;
};

struct kokoro_text_encoder {
    ggml_tensor *                            embd;
    vector<kokoro_text_encoder_conv_layer *> conv_layers;
    lstm *                                   out_lstm;
};

struct kokoro_hparams {
    // standard configuration for Kokoro's Albert model
    // tokenization
    uint32_t bos_token_id         = 0;
    uint32_t eos_token_id         = 0;
    uint32_t space_token_id       = 16;
    // duration prediction
    uint32_t max_context_length   = 512;
    uint32_t vocab_size           = 178;
    uint32_t hidden_size          = 768;
    uint32_t n_attn_heads         = 12;
    uint32_t n_layers             = 1;
    uint32_t n_recurrence         = 12;
    uint32_t head_size            = 64;
    uint32_t duration_hidden_size = 512;
    uint32_t up_sampling_factor;
    float    upsample_scale = 300.0f;
    float    scale          = 0.125f;

    // standard configuration for duration prediction
    uint32_t f0_n_blocks                  = 3;
    uint32_t n_duration_prediction_layers = 3;
    // while it is technically possible for the duration predictor to assign 50 values per token there is no practical need to
    // allocate that many items to the sequence as it is impossible for all tokens to require such long durations and each
    // allocation increases node allocation size by O(N)
    uint32_t max_duration_per_token       = 20;
    uint32_t style_half_size              = 128;

    // standard text encoding configuration
    uint32_t n_conv_layers = 3;

    // standard decoder configuration
    uint32_t n_kernels        = 3;
    uint32_t n_upsamples      = 2;
    uint32_t n_decoder_blocks = 4;
    uint32_t n_res_blocks     = 6;
    uint32_t n_noise_blocks   = 2;
    uint32_t out_conv_padding = 3;
    uint32_t post_n_fft       = 11;
    uint32_t true_n_fft       = 20;
    uint32_t stft_hop         = 5;
    uint32_t harmonic_num     = 8;
    float    sin_amp          = 0.1f;
    float    noise_std        = 0.003f;
    float    voice_threshold  = 10.0f;
    float    sample_rate      = 24000.0f;
    string   window           = "hann";

    // Kokoro loads albert with use_pooling = true but doesn't use the pooling outputs.
    bool uses_pooling       = false;
    bool static_token_types = true;
};

struct albert_layer;

struct kokoro_model : tts_model {
    kokoro_hparams hparams;
    // It is really annoying that ggml doesn't allow using non ggml tensors as the operator for simple math ops.
    // This is just the constant defined above as a tensor.
    ggml_tensor *  n_kernels_tensor;

    map<string, ggml_tensor *> voices;

    // Albert portion of the model
    ggml_tensor *          embd_hidden;
    ggml_tensor *          embd_hidden_bias;
    ggml_tensor *          token_type_embd = nullptr;
    ggml_tensor *          token_embd;
    ggml_tensor *          position_embd;
    ggml_tensor *          input_norm_weight;
    ggml_tensor *          input_norm_bias;
    ggml_tensor *          static_token_type_values = nullptr;
    ggml_tensor *          pool                     = nullptr;
    ggml_tensor *          pool_bias                = nullptr;
    vector<albert_layer *> layers;

    ggml_tensor * harmonic_sampling_norm = nullptr;  // a static 1x9 harmonic multiplier
    ggml_tensor * sampling_factor_scalar = nullptr;  // a static scalar
    ggml_tensor * sqrt_tensor            = nullptr;  // static tensor for constant division

    // Prosody Predictor portion of the model
    duration_predictor * prosody_pred;

    // Text encoding portion of the model
    kokoro_text_encoder * text_encoder;

    // Decoding and Generation portion of the model
    kokoro_decoder * decoder;

    // the default hidden states need to be initialized
    vector<lstm *> lstms;

    size_t   duration_node_counter   = 0;
    size_t   generation_node_counter = 0;
    // setting this is likely unnecessary as it is precomputed by the post load function.
    uint32_t post_load_tensor_bytes  = 13000;

    size_t max_gen_nodes();
    size_t max_duration_nodes();

    lstm * prep_lstm();

    void post_load_assign();
    void assign_weight(std::string name, ggml_tensor * tensor);
    void prep_layers(gguf_context * meta);
    void prep_constants(gguf_context * meta);

    void setup_from_file(gguf_context * meta_ctx, ggml_context * load_context, bool cpu_only = true) {
        std::function<void(ggml_tensor *)> fn = [&](ggml_tensor * cur) {
            const string name      = ggml_get_name(cur);
            size_t       increment = 1;
            if (name.contains("lstm")) {
                increment = max_context_length;
            }
            if (name.contains("duration_predictor")) {
                duration_node_counter += increment;
            } else {
                generation_node_counter += increment;
            }
        };
        compute_tensor_meta_cb                = &fn;
        prep_constants(meta_ctx);
        prep_layers(meta_ctx);
        tts_model::setup_from_file(load_context, cpu_only, "kokoro", 1.6, post_load_tensor_bytes);
    }
};
