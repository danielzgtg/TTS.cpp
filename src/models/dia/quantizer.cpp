#include "quantizers_impl.h"

void dia_register() {}

struct dia_model_quantizer final : tts_model_regex_quantizer {
    // The DAC audio encoder / decoder is not compatible with quantization
    // and normalization tensors should not be quantized.
    explicit dia_model_quantizer() : tts_model_regex_quantizer{ "dia", "^audio_encoder|norm$", "" } {}

    ggml_type get_quantize_type(string_view name, const quantization_params & params) const override {
        if (!params.quantize_output_heads && name.starts_with("dia.decoder.heads")) {
            return GGML_TYPE_F32;
        }
        if (params.convert_dac_to_f16 && name.starts_with("audio_encoder") && name.ends_with("bias")) {
            // To ensure sha256sum matches during testing,
            // partial-revert commit 6642eae1f2051f557eff74394ae24d4936a6e243
            // TODO: until mmwillet/Dia_GGUF is updated
            return GGML_TYPE_F32;
        }
        const ggml_type result = tts_model_regex_quantizer::get_quantize_type(name, params);
        if (result <= GGML_TYPE_F16) {
            return result;
        }
        return result;
    }
} dia_quantizer{};
