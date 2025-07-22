#include "ggml.h"
#include "model.h"

kokoro_generator_upsample_block::kokoro_generator_upsample_block(gguf_context * meta, int index) {
    const string base        = "kokoro.decoder.generator.up_convs." + to_string(index);
    const int    stride_key  = gguf_find_key(meta, (base + ".stride").c_str());
    const int    padding_key = gguf_find_key(meta, (base + ".padding").c_str());
    if (padding_key == -1 || stride_key == -1) {
        TTS_ABORT("both padding and stride keys must be assigned in order to initialize a kokoro upsample block.");
    }
    stride  = gguf_get_val_u32(meta, stride_key);
    padding = gguf_get_val_u32(meta, padding_key);
}
