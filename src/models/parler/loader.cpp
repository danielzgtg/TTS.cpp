#include "../../../include/tokenizer.h"
#include "common.h"
#include "context.h"
#include "decoder/dac/context.h"
#include "decoder/dac/model.h"
#include "kv.h"
#include "model.h"
#include "models/loaders.h"
#include "sampler.h"

void parler_register() {}

struct parler_model_loader final : tts_model_loader {
    explicit parler_model_loader() : tts_model_loader{ "parler-tts" } {}

    unique_ptr<tts_runner> from_file(gguf_context * meta_ctx,  // NOLINT(*-convert-member-functions-to-static)
                                     ggml_context * weight_ctx, int n_threads, bool cpu_only,
                                     const generation_configuration & config) const override {
        parler_tts_model *  model       = new parler_tts_model;
        dac_model *         audio_model = new dac_model;
        unigram_tokenizer * ut          = unigram_tokenizer_from_gguf(meta_ctx);
        ut->initialize_tokenizer();
        model->use_cross_attn = config.use_cross_attn;
        model->setup_from_file(meta_ctx, weight_ctx, cpu_only);
        audio_model->setup_from_file(meta_ctx, weight_ctx, cpu_only);
        sampler *           samp          = new sampler;
        dac_context *       dctx          = build_new_dac_context(audio_model, n_threads, cpu_only);
        dac_runner *        audio_decoder = new dac_runner(audio_model, dctx);
        parler_context *    pctx          = build_new_parler_context(model, n_threads, cpu_only);
        parler_kv_cache *   cache         = new parler_kv_cache;
        parler_tts_runner * runner        = new parler_tts_runner(model, audio_decoder, pctx, ut, samp, cache);
        return runner;
    }
} parler_loader{};
