#include "quantizers_impl.h"

void parler_register() {}

struct parler_model_quantizer final : tts_model_regex_quantizer {
    explicit parler_model_quantizer() :
        tts_model_regex_quantizer{ "parler-tts",
                                   R"(^audio_encoder|norm\.bias$|positional_embed$|text_encoding$|norm\.weight$)",
                                   "" } {}

    ggml_type get_quantize_type(string_view name, const quantization_params & params) const override {
        // the DAC audio encoder / decoder is not compatible with quantization,
        // normalization weight shouldn't be quantized, and the text encoding shouldn't be normalized
        if (!params.quantize_output_heads && name.ends_with("weight.head") ||
            !params.quantize_text_embeddings && name.ends_with("embed_prompts") ||
            !params.quantize_cross_attn_kv &&
                (name.ends_with("encoder_attn.k_proj.weight") || name.ends_with("encoder_attn.v_proj.weight"))) {
            return GGML_TYPE_F32;
        }
        const ggml_type result = tts_model_regex_quantizer::get_quantize_type(name, params);
        if (result <= GGML_TYPE_F16) {
            return result;
        }
        return result;
    }
} parler_quantizer{};
