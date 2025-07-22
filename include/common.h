#pragma once

#include <map>

#include "args_config.h"
#include "imports.h"

// Using this simple struct as opposed to a common std::vector allows us to return the cpu buffer
// pointer directly rather than copying the contents of the buffer to a predefined std::vector.
struct tts_response {
    float *  data;
    size_t   n_outputs = 0;
    // this parameter is only currently used by the t5_encoder for which n_outputs corresponds to sequence length;
    uint32_t hidden_size;
};

struct llama_mmap;
struct tts_model_loader;

struct tts_runner {
    const reference_wrapper<const tts_model_loader> loader;
    unique_ptr<llama_mmap> buf;
    ggml_context *                                  ctx           = nullptr;
    float                                           sampling_rate = 44100.0f;

    explicit tts_runner(const tts_model_loader & loader);

    virtual ~tts_runner() = default;
    virtual void prepare_post_load()                                    = 0;
    virtual void assign_weight(const char * name, ggml_tensor & tensor) = 0;

    virtual void configure_generation(const generation_configuration & config) {
        // TODO merge to generate()
    }

    virtual void generate(const char * prompt, tts_response & response, const generation_configuration & config) = 0;

    void init_build(vector<uint8_t> * buf_compute_meta);
    void free_build();
};
