#include "context.h"
#include "duration/context.h"
#include "loaders.h"
#include "model.h"
#include "phonemizer.h"

void kokoro_register() {}

struct kokoro_model_loader final : tts_model_loader {
    explicit kokoro_model_loader() : tts_model_loader{ "kokoro" } {}

    unique_ptr<tts_model> from_file(gguf_context * meta_ctx,  // NOLINT(*-convert-member-functions-to-static)
                                    ggml_context * weight_ctx, int n_threads, bool cpu_only,
                                    const generation_configuration & config) const override {
        unique_ptr<kokoro_model> model = make_unique<kokoro_model>();
        single_pass_tokenizer *  spt   = single_pass_tokenizer_from_gguf(meta_ctx, "tokenizer.ggml.tokens");
        model->setup_from_file(meta_ctx, weight_ctx, cpu_only);
        kokoro_duration_context * kdctx           = build_new_duration_kokoro_context(&*model, n_threads, cpu_only);
        auto *                    duration_runner = new kokoro_duration_runner(&*model, kdctx, spt);
        kokoro_context *          kctx            = build_new_kokoro_context(&*model, n_threads, cpu_only);
        // if an espeak voice id wasn't specifically set infer it from the kokoro voice,
        // if it was override it, otherwise fallback to American English.
        const char *              espeak_voice_id{ config.espeak_voice_id };
        if (!*espeak_voice_id) {
            espeak_voice_id = get_espeak_id_from_kokoro_voice(config.voice);
        }
        phonemizer * phmzr  = phonemizer_from_gguf(meta_ctx, espeak_voice_id);
        auto *       runner = new kokoro_runner(move(model), kctx, spt, duration_runner, phmzr);
        return runner;
    }
} kokoro_loader{};
