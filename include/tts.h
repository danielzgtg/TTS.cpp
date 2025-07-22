#pragma once

#include "common.h"

unique_ptr<tts_runner> runner_from_file(const char * fname, int n_threads, const generation_configuration & config,
                                        bool cpu_only, bool use_mmap);
void                   generate(tts_runner & runner, const char * prompt, tts_response & response,
                                const generation_configuration & config);
void update_conditional_prompt(tts_runner * runner, const char * file_path, const char * prompt, bool cpu_only);
