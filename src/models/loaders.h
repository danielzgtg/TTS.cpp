#pragma once

#include "common.h"
#include "ggml.h"

struct tts_model_loader {
    /// Installs a model loader for the specified model architecture name
    explicit tts_model_loader(const char * arch);
    const char * const             arch;
    virtual unique_ptr<tts_runner> from_file(gguf_context * meta_ctx, ggml_context * weight_ctx, int n_threads,
                                             bool                                           cpu_only,
                                             /* TODO rm */ const generation_configuration & config) const = 0;
  protected:
    ~tts_model_loader() = default;
};
