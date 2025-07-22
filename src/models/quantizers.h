#pragma once

#include "ggml.h"
#include "imports.h"

struct quantization_params {
    uint32_t  n_threads;
    ggml_type quantize_type;
    bool      quantize_output_heads;
    bool      quantize_text_embeddings;
    bool      quantize_cross_attn_kv;
    bool      convert_dac_to_f16;
    bool      convert_non_quantizable_to_f16;
};

class tts_model_quantizer {
  protected:
    ~tts_model_quantizer() = default;
  public:
    /// Installs a model quantizer for the specified model architecture name
    explicit tts_model_quantizer(const char * arch);
    const char * const arch; /// gguf general.architecture

    /**
     * @param name gguf tensor name
     * @param params user's quantization options
     * @return User's quantization type if possible, or F16 if only that's possible,
     * or F32 to keep unquantizable tensors as-is
     */
    virtual ggml_type get_quantize_type(string_view name, const quantization_params & params) const = 0;
};

const tts_model_quantizer & quantizer_from_gguf(const gguf_context * meta);
