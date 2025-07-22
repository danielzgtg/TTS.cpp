#include "../duration/context.h"
#include "util.h"

ggml_tensor * build_albert_attn_mask(ggml_context * ctx, kokoro_duration_context * kctx, const kokoro_ubatch & batch) {
    kctx->attn_mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, (int64_t) batch.n_tokens, (int64_t) batch.n_tokens);
    ggml_set_input(kctx->attn_mask);

    return kctx->attn_mask;
}

ggml_tensor * build_albert_norm(ggml_context * ctx, ggml_tensor * cur, ggml_tensor * weight, ggml_tensor * bias) {
    // this is the standard eps for Albert
    float eps = 0.000000000001;
    cur       = ggml_norm(ctx, cur, eps);
    cur       = ggml_cont(ctx, ggml_add(ctx, ggml_mul(ctx, cur, weight), bias));
    return cur;
}

ggml_tensor * build_albert_inputs(ggml_context * ctx, kokoro_model * model, ggml_tensor * input_tokens,
                                  ggml_tensor * positions, ggml_tensor * token_types) {
    ggml_tensor * tinpts = ggml_cont(ctx, ggml_get_rows(ctx, model->token_embd, input_tokens));
    ggml_tensor * pinpts = ggml_get_rows(ctx, model->position_embd, positions);

    ggml_tensor * inpts = ggml_cont(ctx, ggml_add(ctx, tinpts, pinpts));
    if (!model->static_token_types) {
        // Token type embeddings are actually static for kokoro at the moment,
        // so we should never need to compute this on the fly.
        return ggml_add(ctx, inpts, ggml_get_rows(ctx, model->token_type_embd, token_types));
    }
    ggml_tensor * ainpts = ggml_add(ctx, inpts, model->static_token_type_values);

    ggml_tensor * out =
        ggml_cont(ctx, build_albert_norm(ctx, ainpts, model->input_norm_weight, model->input_norm_bias));
    return ggml_add(ctx, ggml_mul_mat(ctx, model->embd_hidden, out), model->embd_hidden_bias);
}
