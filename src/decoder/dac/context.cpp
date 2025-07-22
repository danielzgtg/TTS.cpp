#include "context.h"
#include "model.h"

dac_context * build_new_dac_context(dac_model * model, int n_threads, bool use_cpu) {
    dac_context * dctx = new dac_context(model, n_threads);
    if (!use_cpu) {
#ifdef GGML_USE_METAL
        dctx->backend = ggml_backend_metal_init();
#endif
    }
    dctx->backend_cpu = ggml_backend_cpu_init();
    dctx->set_threads();
    dctx->build_schedule(model->max_nodes());
    dctx->buf_compute_meta.resize(ggml_tensor_overhead() * model->max_nodes() +
                                  ggml_graph_overhead_custom(model->max_nodes(), false));
    return dctx;
}

dac_runner::dac_runner(dac_model * model, dac_context * context) : model(model), dctx(context) {}

dac_runner::~dac_runner() {
    if (ctx) {
        ggml_free(ctx);
    }
    delete model;
    delete dctx;
}

void dac_runner::run(uint32_t * input_tokens, uint32_t sequence_length, tts_response * outputs) {
    dac_ubatch batch{input_tokens, sequence_length};
    ggml_backend_sched_reset(dctx->sched);

    const size_t prev_size = dctx->buf_output ? ggml_backend_buffer_get_size(dctx->buf_output) : 0;
    const size_t new_size  = model->max_generation_size * model->up_sampling_factor * sizeof(float);

    if (!dctx->buf_output || prev_size < new_size) {
        if (dctx->buf_output) {
            ggml_backend_buffer_free(dctx->buf_output);
            dctx->buf_output = nullptr;
            dctx->logits     = nullptr;
        }

        dctx->buf_output = ggml_backend_buft_alloc_buffer(dctx->backend_cpu_buffer, new_size);
    }

    outputs->data = (float *) ggml_backend_buffer_get_base(dctx->buf_output);
    ggml_backend_buffer_clear(dctx->buf_output, 0);

    ggml_cgraph * gf = nullptr;
    gf               = build_dac_graph(batch);

    // the output is always the last tensor in the graph
    ggml_tensor * result = gf->nodes[gf->n_nodes - 1];
    ggml_backend_sched_alloc_graph(dctx->sched, gf);

    ggml_backend_tensor_set(dctx->inp_tokens, batch.input_tokens, 0,
                            batch.sequence_length * model->n_heads * ggml_element_size(dctx->inp_tokens));

    ggml_backend_sched_graph_compute_async(dctx->sched, gf);

    dctx->get_ggml_node_data(result, outputs->data, batch.sequence_length * sizeof(float) * model->up_sampling_factor);

    // Reset state for the next token before backend sync, to allow the CPU activities in the reset to
    // overlap with device computation.
    ggml_backend_sched_reset(dctx->sched);
    outputs->n_outputs = sequence_length * model->up_sampling_factor;
}
