#include "quantizers_impl.h"

void kokoro_register() {}

struct kokoro_model_quantizer final : tts_model_regex_quantizer {
    explicit kokoro_model_quantizer() :
        tts_model_regex_quantizer{ "kokoro", "alpha|beta|bias|gamma|voice_tensors|embd$|norm$",
                                   R"(^kokoro\.albert|^kokoro\.text_encoder\.lstm|^kokoro\.duration_predictor\.)"
                                   R"((?:duration_lstm|duration_proj|encode|layers|shared_lstm))" } {}
} kokoro_quantizer{};
