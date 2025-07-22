#pragma once

#include "quantizers.h"

void quantize_gguf(const char * ifile, const char * ofile, const quantization_params & params);
