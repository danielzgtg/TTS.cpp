#include "quantizers.h"

#include <iostream>
#include <unordered_map>

#include "imports.h"

static unordered_map<string_view, reference_wrapper<const tts_model_quantizer>> LOADERS{};

[[maybe_unused]] static bool ld_workaround = [] {
    LD_WORKAROUND
    return true;
}();

tts_model_quantizer::tts_model_quantizer(const char * arch) : arch{ arch } {
    LOADERS.emplace(arch, ref(*this));
}

const tts_model_quantizer & quantizer_from_gguf(const gguf_context * meta) {
    const char * arch{ "parler-tts" };  // only parler-tts gguf files should lack an explicit architecture.
    if (const int arch_key{ gguf_find_key(meta, "general.architecture") }; arch_key != -1) {
        arch = gguf_get_val_str(meta, arch_key);
    }
    const auto found{ LOADERS.find(arch) };
    if (found == LOADERS.end()) {
        TTS_ABORT("Unknown architecture %s\n", arch);
    }
    return found->second.get();
}
