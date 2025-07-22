#pragma once

#include "args.h"
#include "args_config.h"
#include "common.h"

void add_common_args(arg_list & args);

generation_configuration parse_generation_config(const arg_list & args);
unique_ptr<tts_runner>   runner_from_args(const arg_list & args, const generation_configuration & config);

void add_text_encoder_arg(arg_list & args);
