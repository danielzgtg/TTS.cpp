

#include "context.h"

void kokoro_duration_runner::run(kokoro_ubatch & batch) {
    ggml_backend_sched_reset(kctx->sched);

    size_t prev_size = kctx->buf_output ? ggml_backend_buffer_get_size(kctx->buf_output) : 0;
    size_t new_size =
        model->max_context_length * (model->duration_hidden_size + model->style_half_size) * sizeof(float);

    if (!kctx->buf_output || prev_size < new_size) {
        if (kctx->buf_output) {
            ggml_backend_buffer_free(kctx->buf_output);
            kctx->buf_output = nullptr;
            kctx->logits     = nullptr;
        }
        kctx->buf_output = ggml_backend_buft_alloc_buffer(kctx->backend_cpu_buffer, new_size);
    }

    prev_size = kctx->buf_len_output ? ggml_backend_buffer_get_size(kctx->buf_len_output) : 0;
    new_size  = model->max_context_length * sizeof(float);

    if (!kctx->buf_len_output || prev_size < new_size) {
        if (kctx->buf_output) {
            ggml_backend_buffer_free(kctx->buf_len_output);
            kctx->buf_len_output = nullptr;
            kctx->lens           = nullptr;
        }

        kctx->buf_len_output = ggml_backend_buft_alloc_buffer(kctx->backend_cpu_buffer, new_size);
    }

    batch.resp->hidden_states = (float *) ggml_backend_buffer_get_base(kctx->buf_output);
    ggml_backend_buffer_clear(kctx->buf_output, 0);
    batch.resp->lengths = (float *) ggml_backend_buffer_get_base(kctx->buf_len_output);
    ggml_backend_buffer_clear(kctx->buf_len_output, 0);

    ggml_cgraph * gf = NULL;
    gf               = build_kokoro_duration_graph(batch);

    // the output is always the last tensor in the graph
    ggml_tensor * lens          = gf->nodes[gf->n_nodes - 1];
    // the reused duration hidden states are computed before a node chunk which has a size that is sequence length dependent
    ggml_tensor * hidden_states = gf->nodes[gf->n_nodes - 22 - 52 * batch.n_tokens];
    ggml_backend_sched_alloc_graph(kctx->sched, gf);

    set_inputs(batch);

    ggml_backend_sched_graph_compute_async(kctx->sched, gf);

    kctx->get_ggml_node_data(lens, batch.resp->lengths, batch.n_tokens * sizeof(float), kctx->buf_len_output);
    kctx->get_ggml_node_data(hidden_states, batch.resp->hidden_states,
                             batch.n_tokens * (model->duration_hidden_size + model->style_half_size) * sizeof(float));

    // Reset state for the next token before backend sync, to allow the CPU activities in the reset to
    // overlap with device computation.
    ggml_backend_sched_reset(kctx->sched);
    batch.resp->n_outputs = batch.n_tokens;
}

void kokoro_duration_runner::set_inputs(kokoro_ubatch & batch) {
    ggml_backend_tensor_set(kctx->inp_tokens, batch.input_tokens, 0,
                            batch.n_tokens * ggml_element_size(kctx->inp_tokens));
    uint32_t * positions_d = nullptr;
    positions_d            = (uint32_t *) kctx->positions->data;
    float * attn_d         = nullptr;
    attn_d                 = (float *) kctx->attn_mask->data;
    for (uint32_t i = 0; i < batch.n_tokens; i++) {
        positions_d[i] = i;
        for (uint32_t ii = 0; ii < batch.n_tokens; ii++) {
            attn_d[i * batch.n_tokens + ii] =
                0.0f;  // Kokoro doesn't use causal attention as it isnt an autoregressive generative model;
        }
    }
}
