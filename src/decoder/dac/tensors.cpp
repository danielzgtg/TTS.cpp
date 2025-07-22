#include <map>

#include "model.h"

enum dac_tensor {
    DAC_ENCODER_IN_KERNEL,
    DAC_ENCODER_IN_BIAS,
    DAC_ENCODER_OUT_KERNEL,
    DAC_ENCODER_OUT_BIAS,
    DAC_ENCODER_SNAKE_ALPHA,
    DAC_ENCODER_LAYER_SNAKE_ALPHA,
    DAC_ENCODER_LAYER_OUT_KERNEL,
    DAC_ENCODER_LAYER_OUT_BIAS,
    DAC_ENCODER_LAYER_RES_BLK_IN_SNAKE,
    DAC_ENCODER_LAYER_RES_BLK_OUT_SNAKE,
    DAC_ENCODER_LAYER_RES_BLK_IN_KERNEL,
    DAC_ENCODER_LAYER_RES_BLK_OUT_KERNEL,
    DAC_ENCODER_LAYER_RES_BLK_IN_BIAS,
    DAC_ENCODER_LAYER_RES_BLK_OUT_BIAS,
    DAC_QUANTIZER_LAYER_IN_KERNEL,
    DAC_QUANTIZER_LAYER_IN_BIAS,
    DAC_QUANTIZER_LAYER_OUT_KERNEL,
    DAC_QUANTIZER_LAYER_OUT_BIAS,
    DAC_QUANTIZER_LAYER_CODEBOOK
};

static const map<std::string, dac_tensor> DAC_TENSOR_GGUF_LOOKUP = {
    { "initial.bias",        DAC_ENCODER_IN_BIAS                  },
    { "initial.weight",      DAC_ENCODER_IN_KERNEL                },
    { "final.bias",          DAC_ENCODER_OUT_BIAS                 },
    { "final.weight",        DAC_ENCODER_OUT_KERNEL               },
    { "final.alpha",         DAC_ENCODER_SNAKE_ALPHA              },
    { ".final.alpha",        DAC_ENCODER_LAYER_SNAKE_ALPHA        },
    { ".final.bias",         DAC_ENCODER_LAYER_OUT_BIAS           },
    { ".final.weight",       DAC_ENCODER_LAYER_OUT_KERNEL         },
    { ".res.initial.alpha",  DAC_ENCODER_LAYER_RES_BLK_IN_SNAKE   },
    { ".res.initial.bias",   DAC_ENCODER_LAYER_RES_BLK_IN_BIAS    },
    { ".res.initial.weight", DAC_ENCODER_LAYER_RES_BLK_IN_KERNEL  },
    { ".res.final.alpha",    DAC_ENCODER_LAYER_RES_BLK_OUT_SNAKE  },
    { ".res.final.bias",     DAC_ENCODER_LAYER_RES_BLK_OUT_BIAS   },
    { ".res.final.weight",   DAC_ENCODER_LAYER_RES_BLK_OUT_KERNEL },
    { ".in_proj.bias",       DAC_QUANTIZER_LAYER_IN_BIAS          },
    { ".in_proj.weight",     DAC_QUANTIZER_LAYER_IN_KERNEL        },
    { ".out_proj.bias",      DAC_QUANTIZER_LAYER_OUT_BIAS         },
    { ".out_proj.weight",    DAC_QUANTIZER_LAYER_OUT_KERNEL       },
    { ".codebook.weight",    DAC_QUANTIZER_LAYER_CODEBOOK         },
};

void assign_quantizer_layer(dac_model * model, dac_quantize_layer * layer, std::string name, ggml_tensor * tensor) {
    try {
        switch (DAC_TENSOR_GGUF_LOOKUP.at(name)) {
            case DAC_QUANTIZER_LAYER_OUT_KERNEL:
                layer->out_proj_kernel = ggml_dup_tensor(model->ctx, tensor);
                model->set_tensor(layer->out_proj_kernel, tensor);
                break;
            case DAC_QUANTIZER_LAYER_OUT_BIAS:
                layer->out_proj_bias = ggml_dup_tensor(model->ctx, ggml_transpose(model->ctx, tensor));
                model->set_tensor(layer->out_proj_bias, tensor);
                break;
            case DAC_QUANTIZER_LAYER_CODEBOOK:
                layer->codebook = ggml_dup_tensor(model->ctx, tensor);
                model->set_tensor(layer->codebook, tensor);
                break;
            default:
                fprintf(stdout, "quantized layer unassigned tensor %s\n", name.c_str());
                break;
        }
    } catch (const std::out_of_range & e) {
        TTS_ABORT("Error: %s\nTensor, '%s', is not a valid tensor.", e.what(), name.c_str());
    }
}

void assign_residual_unit(dac_model * model, dac_residual_unit * l, string name, ggml_tensor * tensor) {
    try {
        dac_tensor tensor_type = DAC_TENSOR_GGUF_LOOKUP.at(name);
        switch (tensor_type) {
            case DAC_ENCODER_LAYER_RES_BLK_IN_SNAKE:
                l->in_snake_alpha = ggml_dup_tensor(model->ctx, tensor);
                model->set_tensor(l->in_snake_alpha, tensor);
                break;
            case DAC_ENCODER_LAYER_RES_BLK_OUT_SNAKE:
                l->out_snake_alpha = ggml_dup_tensor(model->ctx, tensor);
                model->set_tensor(l->out_snake_alpha, tensor);
                break;
            case DAC_ENCODER_LAYER_RES_BLK_IN_KERNEL:
                l->in_conv_kernel = ggml_dup_tensor(model->ctx, tensor);
                model->set_tensor(l->in_conv_kernel, tensor);
                break;
            case DAC_ENCODER_LAYER_RES_BLK_OUT_KERNEL:
                l->out_conv_kernel = ggml_dup_tensor(model->ctx, tensor);
                model->set_tensor(l->out_conv_kernel, tensor);
                break;
            case DAC_ENCODER_LAYER_RES_BLK_IN_BIAS:
                l->in_conv_bias = ggml_dup_tensor(model->ctx, ggml_transpose(model->ctx, tensor));
                model->set_tensor(l->in_conv_bias, tensor);
                break;
            case DAC_ENCODER_LAYER_RES_BLK_OUT_BIAS:
                l->out_conv_bias = ggml_dup_tensor(model->ctx, ggml_transpose(model->ctx, tensor));
                model->set_tensor(l->out_conv_bias, tensor);
                break;
            default:
                fprintf(stdout, "residual unit unassigned tensor %s\n", name.c_str());
                break;
        }
    } catch (const std::out_of_range & e) {
        TTS_ABORT("Error: %s\nTensor, '%s', is not a valid tensor.", e.what(), name.c_str());
    }
}

void assign_dac_layer(dac_model * model, dac_layer * layer, std::string name, ggml_tensor * tensor) {
    if (DAC_TENSOR_GGUF_LOOKUP.find(name) != DAC_TENSOR_GGUF_LOOKUP.end()) {
        switch (DAC_TENSOR_GGUF_LOOKUP.at(name)) {
            case DAC_ENCODER_LAYER_SNAKE_ALPHA:
                layer->snake_alpha_in = ggml_dup_tensor(model->ctx, tensor);
                model->set_tensor(layer->snake_alpha_in, tensor);
                break;
            case DAC_ENCODER_LAYER_OUT_KERNEL:
                layer->out_conv_kernel = ggml_dup_tensor(model->ctx, tensor);
                model->set_tensor(layer->out_conv_kernel, tensor);
                break;
            case DAC_ENCODER_LAYER_OUT_BIAS:
                layer->out_conv_bias = ggml_dup_tensor(model->ctx, ggml_transpose(model->ctx, tensor));
                model->set_tensor(layer->out_conv_bias, tensor);
                break;
            default:
                fprintf(stdout, "layer unassigned tensor %s\n", name.c_str());
                break;
        }
    } else if (std::find_if(name.begin(), name.end(), ::isdigit) != name.end()) {
        auto        pair    = parse_layer_count(name);
        int         l       = pair.first;
        std::string lt_name = pair.second;
        assign_residual_unit(model, &layer->residual_blocks[l], lt_name, tensor);
    }
}

void assign_to_audio_encoder(dac_model * model, std::string name, ggml_tensor * tensor) {
    if (DAC_TENSOR_GGUF_LOOKUP.find(name) != DAC_TENSOR_GGUF_LOOKUP.end()) {
        switch (DAC_TENSOR_GGUF_LOOKUP.at(name)) {
            case DAC_ENCODER_IN_BIAS:
                model->in_conv_bias = ggml_dup_tensor(model->ctx, ggml_transpose(model->ctx, tensor));
                model->set_tensor(model->in_conv_bias, tensor);
                break;
            case DAC_ENCODER_IN_KERNEL:
                model->in_conv_kernel = ggml_dup_tensor(model->ctx, tensor);
                model->set_tensor(model->in_conv_kernel, tensor);
                break;
            case DAC_ENCODER_OUT_BIAS:
                model->out_conv_bias = ggml_dup_tensor(model->ctx, ggml_transpose(model->ctx, tensor));
                model->set_tensor(model->out_conv_bias, tensor);
                break;
            case DAC_ENCODER_OUT_KERNEL:
                model->out_conv_kernel = ggml_dup_tensor(model->ctx, tensor);
                model->set_tensor(model->out_conv_kernel, tensor);
                break;
            case DAC_ENCODER_SNAKE_ALPHA:
                model->snake_alpha = ggml_dup_tensor(model->ctx, tensor);
                model->set_tensor(model->snake_alpha, tensor);
                break;
            default:
                fprintf(stdout, "unassigned tensor %s\n", name.c_str());
                break;
        }
    } else if (ranges::find_if(name, ::isdigit) != name.end()) {
        auto   pair    = parse_layer_count(name);
        int    l       = pair.first;
        string lt_name = pair.second;
        if (name.find("quantizers") != string::npos) {
            assign_quantizer_layer(model, &model->quantizer_layers[l], lt_name, tensor);
        } else {
            assign_dac_layer(model, &model->layers[l - 1], lt_name, tensor);
        }
    }
}

void dac_model::assign_weight(string name, ggml_tensor * tensor) {
    assign_to_audio_encoder(this, name, tensor);
}
