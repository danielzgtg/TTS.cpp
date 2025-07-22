

#include "ggml.h"

ggml_tensor * dia_layer_norm(ggml_context * ctx, ggml_tensor * inputs, ggml_tensor * weight) {
    // dia always uses 1e-5 as the default eps
    float eps = 0.00001;
    inputs    = ggml_rms_norm(ctx, inputs, eps);
    return ggml_mul(ctx, inputs, weight);
}

/*
 * There are two unique features of Dia's model architecture:
 * 1.  Dia cleans its output generation by adding the difference
 *     between its text based output (its conditional output) and its unconditional output
 *     to the conditional output before sampling. This is why the batch is set to two throughout the graph.
 *
 * 2.  Dia's decoder attends across the entire encoded space
 *     including the pad buffer which receives a unique attention mask.
 *     This is why the encoder sequence is always max length.
 */
ggml_cgraph * dia_runner::build_dia_graph(dia_ubatch & batch) {
    init_build();
    ggml_cgraph * gf             = ggml_new_graph_custom(ctx, 8192, false);
    ggml_tensor * encoded_states = nullptr;

    if (batch.encoder_step) {
        encoded_states = build_dia_encoder(ctx, model, dctx, batch);
        ggml_build_forward_expand(gf, encoded_states);
    }

    ggml_tensor * cur = build_dia_decoder(gf, ctx, model, dctx, kv_cross_self, batch, encoded_states);
    ggml_set_name(cur, "decoder_output");
    ggml_build_forward_expand(gf, cur);
    free_build();

    return gf;
}

dia_ubatch dia_runner::build_worst_case_batch() {
    dia_ubatch batch{ 1, true };
    batch.tokens.resize(model->max_encoder_context_length * 2);
    batch.audio_tokens.resize(model->n_output_heads);
    return batch;
}

void dia_runner::prepare_post_load() {
    dac_runner->prepare_post_load();
    dia_kv_cache_init(kv_cross_self, model, dctx);
    auto batch            = build_worst_case_batch();
    batch.sentence_length = model->max_encoder_context_length;
    dctx->prompt_size     = model->max_encoder_context_length;
    auto gf               = build_dia_graph(batch);
    dctx->prep_schedule(gf);
}
