#include "context.h"
#include "model.h"

namespace {
ggml_tensor * dac_build_audio_inputs(ggml_context * ctx, dac_context * dctx, const dac_ubatch & batch) {
    ggml_tensor * embd;

    dctx->inp_tokens = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, batch.sequence_length*dctx->model->n_heads);
    ggml_set_input(dctx->inp_tokens);

    if (dctx->backend) {
        ggml_backend_sched_set_tensor_backend(dctx->sched, dctx->inp_tokens, dctx->backend);
    }

    for(int i = 0; i < dctx->model->n_heads; i++) {
        auto quantize_layer = dctx->model->quantizer_layers[i];
        ggml_tensor * code = ggml_cont(ctx, ggml_view_2d(ctx, dctx->inp_tokens, 1, batch.sequence_length, dctx->model->n_heads*ggml_type_size(GGML_TYPE_I32), i*ggml_type_size(GGML_TYPE_I32)));
        code = ggml_reshape_1d(ctx, code, batch.sequence_length);
        code = ggml_get_rows(ctx, quantize_layer.codebook, code);
        code = ggml_cont(ctx, ggml_transpose(ctx, code));
        code = ggml_conv_1d(ctx, quantize_layer.out_proj_kernel, code, 1, 0, 1);
        code = ggml_add(ctx, code, quantize_layer.out_proj_bias);

        if (i == 0) {
            embd = code;
        } else {
            embd = ggml_add(ctx, embd, code);
        }
    }
    return embd;
}

ggml_tensor * build_residual_unit(ggml_context * ctx, ggml_tensor * cur, dac_residual_unit & u, int padding, int dilation) {
    ggml_tensor * residual = cur;
    cur = snake_1d(ctx, u.in_snake_alpha, cur);
    cur = ggml_conv_1d(ctx, u.in_conv_kernel, cur, 1, padding, dilation);
    cur = ggml_add(ctx, cur, u.in_conv_bias);
    cur = snake_1d(ctx, u.out_snake_alpha, cur);
    cur = ggml_conv_1d(ctx, u.out_conv_kernel,  cur, 1, 0, 1);
    cur = ggml_add(ctx, cur, u.out_conv_bias);
    return ggml_add(ctx, cur, residual);
}

ggml_tensor * build_decoder_block(ggml_context * ctx, ggml_tensor * cur, dac_layer & l) {
    cur = snake_1d(ctx, l.snake_alpha_in, cur);
    cur = ggml_conv_transpose_1d(ctx, l.out_conv_kernel, cur, l.stride, l.padding, 1, 0, 1);
    cur = ggml_add(ctx, cur, l.out_conv_bias);
    for (int i = 0; i < l.residual_blocks.size(); i++) {
        cur = build_residual_unit(ctx, cur, l.residual_blocks[i], pow(3, (i + 1)), pow(3, i));
    }
    return cur;
}
}

ggml_cgraph * dac_runner::build_dac_graph(dac_ubatch & batch) {
    init_build();
    // splitting this out from the primary graph so that we can better manage streaming (i.e. sentence chunks are better performed this way)
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 8192, false);

    ggml_tensor * cur;
    ggml_tensor * inputs;

    inputs = dac_build_audio_inputs(ctx, dctx, batch);
    ggml_set_name(inputs, "quanitzed_inputs");

    // everything besides the inputs is just a forward pass
    cur = ggml_conv_1d(ctx, model->in_conv_kernel, inputs, 1, 3, 1);
    cur = ggml_add(ctx, cur, model->in_conv_bias);
    for (auto l : model->layers) {
        cur = build_decoder_block(ctx, cur, l);
    }
    cur = snake_1d(ctx, model->snake_alpha, cur);
    cur = ggml_conv_1d(ctx, model->out_conv_kernel, cur, 1, 3, 1);
    cur = ggml_add(ctx, cur, model->out_conv_bias);
    cur = ggml_tanh(ctx, cur);
    ggml_build_forward_expand(gf, cur);
    free_build();
    return gf;
}

void dac_runner::prepare_post_load() {
    dac_ubatch batch;
    batch.sequence_length = model->max_generation_size;
    ggml_cgraph * gf = build_dac_graph(batch);
    dctx->prep_schedule(gf);
}
