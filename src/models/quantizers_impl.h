#pragma once

#include <regex>

#include "quantizers.h"

class tts_model_regex_quantizer : tts_model_quantizer {
    // TODO move to 4 vectors instead of regex, which is slow and bloaty
    const regex exclude;
    const regex include;
  protected:
    ~tts_model_regex_quantizer() = default;
  public:
    explicit tts_model_regex_quantizer(const char * arch, const char * exclude_regex, const char * include_regex) :
        tts_model_quantizer{ arch },
        exclude{ exclude_regex },
        include{ include_regex } {}

    ggml_type get_quantize_type(string_view name, const quantization_params & params) const override {
        if (params.convert_dac_to_f16 && name.starts_with("audio_encoder") && !name.ends_with("alpha")) {
            return GGML_TYPE_F16;
        }
        if (regex_search(cbegin(name), cend(name), exclude)) {
            return GGML_TYPE_F32;
        }
        if (!regex_search(cbegin(name), cend(name), include)) {
            return params.convert_non_quantizable_to_f16 ? GGML_TYPE_F16 : GGML_TYPE_F32;
        }
        return params.quantize_type;
    }
};
