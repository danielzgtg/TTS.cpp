#pragma once

#include "buffers.h"
#include "imports.h"

struct ggml_tensor;

struct lstm_cell {
    vector<ggml_tensor *> weights;
    vector<ggml_tensor *> biases;
    vector<ggml_tensor *> reverse_weights;
    vector<ggml_tensor *> reverse_biases;
};

struct lstm {
    vector<ggml_tensor *> hidden;
    vector<ggml_tensor *> states;

    bool                bidirectional = false;
    vector<lstm_cell *> cells;
    ggml_tensor *       build(ggml_context * ctx, ggml_tensor * input, uint32_t sequence_length) const;
    void                assign(tts_buffer & ctx, const string & name, ggml_tensor * tensor) const;
};
