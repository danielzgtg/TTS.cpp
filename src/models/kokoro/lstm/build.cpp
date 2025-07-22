#include "model.h"
#import "util.h"

static ggml_tensor * build_lstm_run(ggml_context * ctx, ggml_tensor * input, ggml_tensor * h_0, ggml_tensor * c_0,
                                    const vector<ggml_tensor *> & weights, const vector<ggml_tensor *> & biases,
                                    uint32_t sequence_length, bool reversed) {
    ggml_tensor * I = ggml_add(ctx, ggml_mul_mat(ctx, weights[0], input), biases[0]);
    ggml_tensor * F = ggml_add(ctx, ggml_mul_mat(ctx, weights[2], input), biases[2]);
    ggml_tensor * G = ggml_add(ctx, ggml_mul_mat(ctx, weights[4], input), biases[4]);
    ggml_tensor * O = ggml_add(ctx, ggml_mul_mat(ctx, weights[6], input), biases[6]);

    ggml_tensor * outputs{};

    for (int index = 0; index < sequence_length; index++) {
        const int     i     = reversed ? sequence_length - 1 - index : index;
        ggml_tensor * I_cur = ggml_view_3d(ctx, I, I->ne[0], 1, I->ne[2], I->nb[0], I->nb[1], I->nb[1] * i);
        I_cur = ggml_sigmoid(ctx, ggml_add(ctx, I_cur, ggml_add(ctx, ggml_mul_mat(ctx, weights[1], h_0), biases[1])));

        ggml_tensor * F_cur = ggml_view_3d(ctx, F, F->ne[0], 1, F->ne[2], F->nb[0], F->nb[1], F->nb[1] * i);
        F_cur = ggml_sigmoid(ctx, ggml_add(ctx, F_cur, ggml_add(ctx, ggml_mul_mat(ctx, weights[3], h_0), biases[3])));

        ggml_tensor * G_cur = ggml_view_3d(ctx, G, G->ne[0], 1, G->ne[2], G->nb[0], G->nb[1], G->nb[1] * i);
        G_cur = ggml_tanh(ctx, ggml_add(ctx, G_cur, ggml_add(ctx, ggml_mul_mat(ctx, weights[5], h_0), biases[5])));

        ggml_tensor * O_cur = ggml_view_3d(ctx, O, O->ne[0], 1, O->ne[2], O->nb[0], O->nb[1], O->nb[1] * i);
        O_cur = ggml_sigmoid(ctx, ggml_add(ctx, O_cur, ggml_add(ctx, ggml_mul_mat(ctx, weights[7], h_0), biases[7])));

        c_0 = ggml_add(ctx, ggml_mul(ctx, F_cur, c_0), ggml_mul(ctx, I_cur, G_cur));
        h_0 = ggml_mul(ctx, ggml_tanh(ctx, c_0), O_cur);

        if (index == 0) {
            outputs = h_0;
        } else {
            outputs = reversed ? ggml_concat(ctx, h_0, outputs, 1) : ggml_concat(ctx, outputs, h_0, 1);
        }
    }
    return outputs;
}

ggml_tensor * lstm::build(ggml_context * ctx, ggml_tensor * input, uint32_t sequence_length) const {
    ggml_tensor * resp         = input;
    ggml_tensor * reverse_resp = input;

    // Iterate over cells first so that at each pass to the next cell,
    // we have a fully formed vector (this improves performance as well as allocation for stacked LSTMs)
    for (int c = 0; c < cells.size(); c++) {
        resp = build_lstm_run(ctx, resp, hidden[c], states[c], cells[c]->weights, cells[c]->biases,
                              sequence_length, false);
        if (bidirectional) {
            reverse_resp =
                build_lstm_run(ctx, reverse_resp, hidden[c], states[c], cells[c]->reverse_weights,
                               cells[c]->reverse_biases, sequence_length, true);
        }
    }
    if (bidirectional) {
        resp = ggml_concat(ctx, resp, reverse_resp, 0);
    }
    return resp;
}
