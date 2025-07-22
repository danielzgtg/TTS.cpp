
#include "ggml.h"


static kokoro_generator_residual_block * build_res_block_from_file(gguf_context * meta, std::string base_config_key) {
    kokoro_generator_residual_block * grb = new kokoro_generator_residual_block;
    // these residual blocks always have 3 convolutional layers
    for (int i = 0; i < 3; i++) {
        grb->adain1d_1_gamma_weights.push_back(nullptr);
        grb->adain1d_2_gamma_weights.push_back(nullptr);
        grb->adain1d_1_gamma_biases.push_back(nullptr);
        grb->adain1d_2_gamma_biases.push_back(nullptr);
        grb->adain1d_1_beta_weights.push_back(nullptr);
        grb->adain1d_2_beta_weights.push_back(nullptr);
        grb->adain1d_1_beta_biases.push_back(nullptr);
        grb->adain1d_2_beta_biases.push_back(nullptr);
        grb->input_alphas.push_back(nullptr);
        grb->output_alphas.push_back(nullptr);
        grb->convs1_weights.push_back(nullptr);
        grb->convs1_biases.push_back(nullptr);
        grb->convs2_weights.push_back(nullptr);
        grb->convs2_biases.push_back(nullptr);
        int padding_key  = gguf_find_key(meta, (base_config_key + "." + std::to_string(i) + ".padding").c_str());
        int dilation_key = gguf_find_key(meta, (base_config_key + "." + std::to_string(i) + ".dilation").c_str());
        if (padding_key == -1 || dilation_key == -1) {
            TTS_ABORT("Could not find dilation and padding for generator residual block at key, '%s.%d'.",
                      base_config_key.c_str(), i);
        }
        grb->conv1_dilations.push_back(gguf_get_val_u32(meta, dilation_key));
        grb->conv1_paddings.push_back(gguf_get_val_u32(meta, padding_key));
    }
    return grb;
}


static kokoro_noise_residual_block * build_noise_block_from_file(gguf_context * meta, int index) {
    kokoro_noise_residual_block * nb   = new kokoro_noise_residual_block;
    std::string                   base = "kokoro.decoder.generator.noise_blocks." + std::to_string(index);
    nb->res_block                      = build_res_block_from_file(meta, base + ".res_block");
    int stride_key                     = gguf_find_key(meta, (base + ".stride").c_str());
    int padding_key                    = gguf_find_key(meta, (base + ".padding").c_str());
    if (padding_key == -1 || stride_key == -1) {
        TTS_ABORT("both padding and stride keys must be assigned in order to initialize a kokoro noise block.");
    }
    nb->input_conv_stride  = gguf_get_val_u32(meta, stride_key);
    nb->input_conv_padding = gguf_get_val_u32(meta, padding_key);
    return nb;
}


lstm * kokoro_model::prep_lstm() {
    lstm *      rnn  = new lstm;
    lstm_cell * cell = new lstm_cell;
    for (int i = 0; i < 8; i++) {
        cell->weights.push_back(nullptr);
        cell->biases.push_back(nullptr);
        cell->reverse_weights.push_back(nullptr);
        cell->reverse_biases.push_back(nullptr);
    }
    rnn->cells.push_back(cell);
    rnn->bidirectional = true;
    lstms.push_back(rnn);
    return rnn;
}

void kokoro_model::prep_layers(gguf_context * meta) {
    prosody_pred                     = new duration_predictor;
    prosody_pred->shared_lstm        = prep_lstm();
    prosody_pred->duration_proj_lstm = prep_lstm();
    text_encoder                     = new kokoro_text_encoder;
    decoder                          = new kokoro_decoder;
    decoder->generator               = new kokoro_generator;
    decoder->encoder_block           = new ada_residual_conv_block;
    text_encoder->out_lstm           = prep_lstm();

    for (int i = 0; i < n_layers; i++) {
        layers.push_back(new albert_layer);
    }

    for (int i = 0; i < f0_n_blocks; i++) {
        ada_residual_conv_block * f0 = new ada_residual_conv_block;
        ada_residual_conv_block * n  = new ada_residual_conv_block;
        prosody_pred->f0_blocks.push_back(f0);
        prosody_pred->n_blocks.push_back(n);
    }

    for (int i = 0; i < n_duration_prediction_layers; i++) {
        duration_predictor_layer * dpl = new duration_predictor_layer;
        dpl->rnn                       = prep_lstm();
        prosody_pred->layers.push_back(dpl);
    }

    for (int i = 0; i < n_decoder_blocks; i++) {
        decoder->decoder_blocks.push_back(new ada_residual_conv_block);
    }

    for (int i = 0; i < n_noise_blocks; i++) {
        kokoro_noise_residual_block * nb = build_noise_block_from_file(meta, i);
        decoder->generator->noise_blocks.push_back(nb);
    }

    for (int i = 0; i < n_upsamples; i++) {
        kokoro_generator_upsample_block * ub = kokoro_generator_upsample_block(meta, i);
        decoder->generator->ups.push_back(ub);
    }

    for (int i = 0; i < n_res_blocks; i++) {
        kokoro_generator_residual_block * rb =
            build_res_block_from_file(meta, "kokoro.decoder.generator.res_blocks." + std::to_string(i));
        decoder->generator->res_blocks.push_back(rb);
    }

    for (int i = 0; i < n_conv_layers; i++) {
        text_encoder->conv_layers.push_back(new kokoro_text_encoder_conv_layer);
    }
}

void kokoro_model::prep_constants(gguf_context * meta) {
    // get constants for the Albert duration prediction model
    int context_size_key = gguf_find_key(meta, "kokoro.duration_predictor.albert.context_length");
    if (context_size_key != -1) {
        max_context_length = gguf_get_val_u32(meta, context_size_key);
    }

    int vocab_size_key = gguf_find_key(meta, "kokoro.tokenizer.vocab_size");
    if (vocab_size_key != -1) {
        vocab_size = gguf_get_val_u32(meta, vocab_size_key);
    }

    int hidden_size_key = gguf_find_key(meta, "kokoro.duration_predictor.albert.hidden_size");
    if (hidden_size_key != -1) {
        hidden_size = gguf_get_val_u32(meta, hidden_size_key);
    }

    int attn_heads_key = gguf_find_key(meta, "kokoro.duration_predictor.albert.attn_heads");
    if (attn_heads_key != -1) {
        n_attn_heads = gguf_get_val_u32(meta, attn_heads_key);
        head_size    = (uint32_t) hidden_size / n_attn_heads;
    }

    int albert_layers_key = gguf_find_key(meta, "kokoro.duration_predictor.albert.layers");
    if (albert_layers_key != -1) {
        n_layers = gguf_get_val_u32(meta, albert_layers_key);
    }

    int recurrence_key = gguf_find_key(meta, "kokoro.duration_predictor.albert.recurrence");
    if (recurrence_key != -1) {
        n_recurrence = gguf_get_val_u32(meta, recurrence_key);
    }

    int duration_hidden_key = gguf_find_key(meta, "kokoro.duration_predictor.hidden_size");
    if (duration_hidden_key != -1) {
        duration_hidden_size = gguf_get_val_u32(meta, duration_hidden_key);
    }

    int up_sampling_factor_key = gguf_find_key(meta, "kokoro.decoder.generator.up_sampling_factor");
    if (up_sampling_factor_key != -1) {
        up_sampling_factor = gguf_get_val_u32(meta, up_sampling_factor_key);
    }

    int f0_n_blocks_key = gguf_find_key(meta, "kokoro.duration_predictor.f0_n_blocks");
    if (f0_n_blocks_key != -1) {
        f0_n_blocks = gguf_get_val_u32(meta, f0_n_blocks_key);
    }

    int duration_pred_layers_key = gguf_find_key(meta, "kokoro.duration_predictor.layers");
    if (duration_pred_layers_key != -1) {
        n_duration_prediction_layers = gguf_get_val_u32(meta, duration_pred_layers_key);
    }

    // get text and decoding configuration for generation
    int n_conv_layers_key = gguf_find_key(meta, "kokoro.text_encoder.layers");
    if (n_conv_layers_key != -1) {
        n_conv_layers = gguf_get_val_u32(meta, n_conv_layers_key);
    }

    int n_kernels_key = gguf_find_key(meta, "kokoro.decoder.generator.kernels");
    if (n_kernels_key != -1) {
        n_kernels = gguf_get_val_u32(meta, n_kernels_key);
    }

    int n_upsamples_key = gguf_find_key(meta, "kokoro.decoder.generator.upsamples");
    if (n_upsamples_key != -1) {
        n_upsamples = gguf_get_val_u32(meta, n_upsamples_key);
    }

    int n_decoder_blocks_key = gguf_find_key(meta, "kokoro.decoder.generator.layers");
    if (n_decoder_blocks_key != -1) {
        n_decoder_blocks = gguf_get_val_u32(meta, n_decoder_blocks_key);
    }

    int out_conv_padding_key = gguf_find_key(meta, "kokoro.decoder.generator.padding");
    if (out_conv_padding_key != -1) {
        out_conv_padding = gguf_get_val_u32(meta, out_conv_padding_key);
    }

    int n_fft_key = gguf_find_key(meta, "kokoro.decoder.generator.n_fft");
    if (n_fft_key != -1) {
        true_n_fft = gguf_get_val_u32(meta, n_fft_key);
        post_n_fft = (uint32_t) true_n_fft / 2 + 1;
    }

    int stft_hop_key = gguf_find_key(meta, "kokoro.decoder.generator.hop");
    if (stft_hop_key != -1) {
        stft_hop = gguf_get_val_u32(meta, stft_hop_key);
    }
}
