#include "tts.h"

#include "parler/context.h"

void generate(tts_runner & runner, const char * prompt, tts_response & response,
              const generation_configuration & config) {
    runner.configure_generation(config);
    runner.generate(prompt, response, config);
}

void update_conditional_prompt(tts_runner * runner, const char * file_path, const char * prompt, bool cpu_only) {
    // TODO STOPSHIP uncomment
    // const auto parler{ dynamic_cast<parler_tts_runner *>(runner) };
    // if (!parler) {
    //     fprintf(stderr, "Wrong model for conditional prompt\n");
    //     return;
    // }
    // parler->update_conditional_prompt(file_path, prompt, parler->pctx->n_threads, cpu_only);
}
