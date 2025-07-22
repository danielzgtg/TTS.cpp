#include "context.h"
#include "util.h"

void kokoro_runner::prepare_post_load() {
    model->post_load_assign();
    drunner->prepare_post_load();
    auto batch = build_worst_case_batch();
    auto gf    = build_kokoro_graph(batch);
    kctx->prep_schedule(gf);
    delete batch.resp;
}

kokoro_ubatch kokoro_runner::build_worst_case_batch() {
    kokoro_ubatch batch;
    batch.n_tokens        = model->max_context_length;
    batch.resp            = new kokoro_duration_response;
    batch.resp->n_outputs = model->max_context_length;
    kctx->total_duration  = model->max_context_length * model->max_duration_per_token;
    kctx->sequence_length = model->max_context_length;
    std::vector<float> lengths;
    lengths.reserve(model->max_context_length);
    for (int i = 0; i < model->max_context_length; i++) {
        lengths.push_back(50.0f);
    }
    batch.resp->lengths = lengths.data();
    return batch;
}

ggml_cgraph * kokoro_runner::build_kokoro_graph(kokoro_ubatch & batch) {
    init_build();
    // This '570000' number is coming from the number of nodes necessary for the longest possible sequence computed by the graph.
    // While it may be possible to precompute this by determining the longest possible duration against he maximum context length of the model,
    // it is not easily performed given that nodes do not necessarily line up predictably with the number of tensors in the model or its submodels.
    // In order to side step this problem I computed the graph and determined the size in advance and use that constant value here.
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 570000, false);

    ggml_tensor * voice      = model->voices[kctx->voice];
    ggml_tensor * style_half = ggml_view_1d(ctx, voice, voice->ne[0] / 2,
                                            voice->ne[0] / 2 * voice->nb[0] + (batch.n_tokens - 3) * voice->nb[1]);
    ggml_tensor * cur;

    kctx->inp_tokens = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, batch.n_tokens);
    ggml_set_input(kctx->inp_tokens);

    kctx->duration_mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kctx->total_duration, kctx->sequence_length);
    ggml_set_input(kctx->duration_mask);

    kctx->duration_pred = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, model->duration_hidden_size + model->style_half_size,
                                             kctx->sequence_length);
    ggml_set_input(kctx->duration_pred);

    // seeing as we are setting the inputs for these, we shouldn't need to perform transpositions here
    cur = ggml_mul_mat(ctx, ggml_cont(ctx, ggml_transpose(ctx, kctx->duration_mask)),
                       ggml_cont(ctx, ggml_transpose(ctx, kctx->duration_pred)));
    cur = ggml_cont(ctx, ggml_transpose(ctx, cur));

    cur = build_lstm(ctx, cur, model->prosody_pred->shared_lstm, cur->ne[1]);

    ggml_tensor * f0_curve = cur;
    f0_curve               = ggml_cont(ctx, ggml_transpose(ctx, f0_curve));
    for (auto block : model->prosody_pred->f0_blocks) {
        f0_curve = build_ada_residual_conv(ctx, f0_curve, block, style_half, model->sqrt_tensor);
    }
    f0_curve = ggml_cont(ctx, ggml_transpose(ctx, f0_curve));
    f0_curve = ggml_mul_mat(ctx, model->prosody_pred->f0_proj_kernel, f0_curve);
    f0_curve = squeeze_3d_2d_e0(ctx, f0_curve);
    f0_curve = ggml_add(ctx, f0_curve, model->prosody_pred->f0_proj_bias);
    ggml_set_name(f0_curve, "f0_out");

    ggml_tensor * n = cur;
    n               = ggml_cont(ctx, ggml_transpose(ctx, n));
    for (auto block : model->prosody_pred->n_blocks) {
        n = build_ada_residual_conv(ctx, n, block, style_half, model->sqrt_tensor);
    }
    n = ggml_cont(ctx, ggml_transpose(ctx, n));
    n = ggml_mul_mat(ctx, model->prosody_pred->n_proj_kernel, n);
    n = squeeze_3d_2d_e0(ctx, n);
    n = ggml_add(ctx, n, model->prosody_pred->n_proj_bias);
    ggml_set_name(n, "n_out");
    ggml_build_forward_expand(gf, n);

    // kokoro text encoding;
    ggml_tensor * asr;
    // ggml_tensor * embd;
    {
        asr = ggml_get_rows(ctx, model->text_encoder->embd, kctx->inp_tokens);

        for (auto l : model->text_encoder->conv_layers) {
            asr = ggml_cont(
                ctx, ggml_transpose(ctx, ggml_add(ctx,
                                                  ggml_conv_1d(ctx, l->conv_weight,
                                                               ggml_cont(ctx, ggml_transpose(ctx, asr)), 1, 2, 1),
                                                  l->conv_bias)));
            asr = ggml_norm(ctx, asr, 0.00001);
            asr = ggml_add(ctx, ggml_mul(ctx, asr, l->norm_gamma), l->norm_beta);
            asr = ggml_leaky_relu(ctx, asr, 0.2f, false);
        }

        asr = build_lstm(ctx, asr, model->text_encoder->out_lstm, kctx->sequence_length);
        asr = ggml_mul_mat(ctx, ggml_cont(ctx, ggml_transpose(ctx, asr)),
                           ggml_cont(ctx, ggml_transpose(ctx, kctx->duration_mask)));
    }

    // decoding and generation prep
    ggml_tensor * asr_res;
    ggml_tensor * f0;
    ggml_tensor * n_base;

    ggml_tensor * style_half2 = ggml_view_1d(ctx, voice, voice->ne[0] / 2, (batch.n_tokens - 3) * voice->nb[1]);

    {
        f0 = ggml_add(ctx, ggml_conv_1d(ctx, model->decoder->f0_conv, f0_curve, 2, 1, 1), model->decoder->f0_conv_bias);
        n_base = ggml_add(ctx, ggml_conv_1d(ctx, model->decoder->n_conv, n, 2, 1, 1), model->decoder->n_conv_bias);
        cur    = ggml_concat(ctx, ggml_concat(ctx, ggml_cont(ctx, ggml_transpose(ctx, asr)), f0, 1), n_base, 1);
        cur    = build_ada_residual_conv(ctx, cur, model->decoder->encoder_block, style_half2, model->sqrt_tensor);

        asr_res = ggml_mul_mat(ctx, model->decoder->asr_conv, asr);
        asr_res = ggml_add(ctx, asr_res, ggml_transpose(ctx, model->decoder->asr_conv_bias));

        asr_res = ggml_cont(ctx, ggml_transpose(ctx, asr_res));
        for (auto l : model->decoder->decoder_blocks) {
            cur = ggml_concat(ctx, ggml_concat(ctx, ggml_concat(ctx, cur, asr_res, 1), f0, 1), n_base, 1);
            cur = build_ada_residual_conv(ctx, cur, l, style_half2, model->sqrt_tensor);
        }
        cur = ggml_cont(ctx, ggml_transpose(ctx, cur));
    }

    kctx->window_sq_sum = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, kctx->total_duration * model->up_sampling_factor);
    ggml_set_input(kctx->window_sq_sum);

    // run generation
    cur = build_generator(ctx, &*model, kctx, cur, style_half2, f0_curve, model->decoder->generator,
                          (int) kctx->sequence_length, kctx->window_sq_sum, gf);
    ggml_build_forward_expand(gf, cur);
    free_build();
    return gf;
}


void kokoro_duration_runner::prepare_post_load() {
    auto batch = build_worst_case_batch();
    auto gf    = build_kokoro_duration_graph(batch);
    kctx->prep_schedule(gf);
}
