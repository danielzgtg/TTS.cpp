
#include "buffers.h"
#include "model.h"
#include "util.h"

void lstm::assign(tts_buffer & ctx, const string & name, ggml_tensor * tensor) const {
    const vector<string> parts = split(name, ".");
    const int            i     = stoi(parts[0]);
    const int            ii    = stoi(parts[2]);
    if (parts[1] == "weights") {
        cells[i]->weights[ii] = ggml_dup_tensor(ctx, tensor);
        ctx.set_tensor(cells[i]->weights[ii], tensor);
    } else if (parts[1] == "biases") {
        cells[i]->biases[ii] = ggml_dup_tensor(ctx, tensor);
        ctx.set_tensor(cells[i]->biases[ii], tensor);
    } else if (parts[1] == "reverse_weights") {
        cells[i]->reverse_weights[ii] = ggml_dup_tensor(ctx, tensor);
        ctx.set_tensor(cells[i]->reverse_weights[ii], tensor);
    } else if (parts[1] == "reverse_biases") {
        cells[i]->reverse_biases[ii] = ggml_dup_tensor(ctx, tensor);
        ctx.set_tensor(cells[i]->reverse_biases[ii], tensor);
    }
}
