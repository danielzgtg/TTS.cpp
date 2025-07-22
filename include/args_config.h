#pragma once

struct generation_configuration {
    bool use_cross_attn{
        true
    };  // TODO split out this load-time option from the rest of the generate-time configuration
    float                 temperature{ 1.0f };
    float                 repetition_penalty{ 1.0f };
    float                 top_p{ 1.0f };
    int                   top_k{ 50 };
    int                   max_tokens{ 0 };
    const char *          voice{ "af_alloy" };
    static constexpr bool sample{ true };
    const char *          espeak_voice_id{ "gmw/en-US" };
};
