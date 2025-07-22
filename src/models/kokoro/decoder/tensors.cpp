
#include "buffers.h"
#include "model.h"
#include "util.h"


void kokoro_model::assign_generator_weight(kokoro_generator * generator, std::string name, ggml_tensor * tensor) {
    if (name == "m_source_weight") {
        generator->m_source_weight = ggml_dup_tensor(ctx, tensor);
        set_tensor(generator->m_source_weight, tensor);
    } else if (name == "m_source_bias") {
        generator->m_source_bias = ggml_dup_tensor(ctx, tensor);
        set_tensor(generator->m_source_bias, tensor);
    } else if (name == "conv_post_weight") {
        generator->out_conv_weight = ggml_dup_tensor(ctx, tensor);
        set_tensor(generator->out_conv_weight, tensor);
    } else if (name == "conv_post_bias") {
        generator->out_conv_bias = ggml_dup_tensor(ctx, ggml_transpose(ctx, tensor));
        set_tensor(generator->out_conv_bias, tensor);
    } else {
        std::vector<std::string> parts = split(name, ".");
        int                      i     = std::stoi(parts[1]);
        if (parts[0] == "noise_blocks") {
            if (parts[2] == "conv_weight") {
                generator->noise_blocks[i]->input_conv = ggml_dup_tensor(ctx, tensor);
                set_tensor(generator->noise_blocks[i]->input_conv, tensor);
            } else if (parts[2] == "conv_bias") {
                generator->noise_blocks[i]->input_conv_bias = ggml_dup_tensor(ctx, ggml_transpose(ctx, tensor));
                set_tensor(generator->noise_blocks[i]->input_conv_bias, tensor);
            } else if (parts[2] == "resblock") {
                assign_gen_resblock(generator->noise_blocks[i]->res_block,
                                    name.substr(parts[0].size() + parts[1].size() + parts[2].size() + 3), tensor);
            }
        } else if (parts[0] == "resblocks") {
            assign_gen_resblock(generator->res_blocks[i], name.substr(parts[0].size() + parts[1].size() + 2), tensor);
        } else if (parts[0] == "ups") {
            if (parts[2] == "weight") {
                generator->ups[i]->upsample_weight = ggml_dup_tensor(ctx, tensor);
                set_tensor(generator->ups[i]->upsample_weight, tensor);
            } else if (parts[2] == "bias") {
                generator->ups[i]->upsample_bias = ggml_dup_tensor(ctx, ggml_transpose(ctx, tensor));
                set_tensor(generator->ups[i]->upsample_bias, tensor);
            }
        }
    }
}

void kokoro_model::assign_gen_resblock(kokoro_generator_residual_block * block, std::string name,
                                       ggml_tensor * tensor) {
    std::vector<std::string> parts = split(name, ".");
    int                      i     = std::stoi(parts[0]);
    if (parts[1] == "gamma1_weight") {
        block->adain1d_1_gamma_weights[i] = ggml_dup_tensor(ctx, tensor);
        set_tensor(block->adain1d_1_gamma_weights[i], tensor);
    } else if (parts[1] == "gamma2_weight") {
        block->adain1d_2_gamma_weights[i] = ggml_dup_tensor(ctx, tensor);
        set_tensor(block->adain1d_2_gamma_weights[i], tensor);
    } else if (parts[1] == "gamma1_bias") {
        block->adain1d_1_gamma_biases[i] = ggml_dup_tensor(ctx, tensor);
        set_tensor(block->adain1d_1_gamma_biases[i], tensor);
    } else if (parts[1] == "gamma2_bias") {
        block->adain1d_2_gamma_biases[i] = ggml_dup_tensor(ctx, tensor);
        set_tensor(block->adain1d_2_gamma_biases[i], tensor);
    } else if (parts[1] == "beta1_weight") {
        block->adain1d_1_beta_weights[i] = ggml_dup_tensor(ctx, tensor);
        set_tensor(block->adain1d_1_beta_weights[i], tensor);
    } else if (parts[1] == "beta2_weight") {
        block->adain1d_2_beta_weights[i] = ggml_dup_tensor(ctx, tensor);
        set_tensor(block->adain1d_2_beta_weights[i], tensor);
    } else if (parts[1] == "beta1_bias") {
        block->adain1d_1_beta_biases[i] = ggml_dup_tensor(ctx, tensor);
        set_tensor(block->adain1d_1_beta_biases[i], tensor);
    } else if (parts[1] == "beta2_bias") {
        block->adain1d_2_beta_biases[i] = ggml_dup_tensor(ctx, tensor);
        set_tensor(block->adain1d_2_beta_biases[i], tensor);
    } else if (parts[1] == "convs1_weight") {
        block->convs1_weights[i] = ggml_dup_tensor(ctx, tensor);
        set_tensor(block->convs1_weights[i], tensor);
    } else if (parts[1] == "convs2_weight") {
        block->convs2_weights[i] = ggml_dup_tensor(ctx, tensor);
        set_tensor(block->convs2_weights[i], tensor);
    } else if (parts[1] == "convs1_bias") {
        block->convs1_biases[i] = ggml_dup_tensor(ctx, ggml_transpose(ctx, tensor));
        set_tensor(block->convs1_biases[i], tensor);
    } else if (parts[1] == "convs2_bias") {
        block->convs2_biases[i] = ggml_dup_tensor(ctx, ggml_transpose(ctx, tensor));
        set_tensor(block->convs2_biases[i], tensor);
    } else if (parts[1] == "alpha1") {
        block->input_alphas[i] = ggml_dup_tensor(ctx, tensor);
        set_tensor(block->input_alphas[i], tensor);
    } else if (parts[1] == "alpha2") {
        block->output_alphas[i] = ggml_dup_tensor(ctx, tensor);
        set_tensor(block->output_alphas[i], tensor);
    }
}

void kokoro_model::assign_decoder_weight(std::string name, ggml_tensor * tensor) {
    if (name == "f0_conv_weight") {
        decoder->f0_conv = ggml_dup_tensor(ctx, tensor);
        set_tensor(decoder->f0_conv, tensor);
    } else if (name == "f0_conv_bias") {
        decoder->f0_conv_bias = ggml_dup_tensor(ctx, ggml_transpose(ctx, tensor));
        set_tensor(decoder->f0_conv_bias, tensor);
    } else if (name == "n_conv_weight") {
        decoder->n_conv = ggml_dup_tensor(ctx, tensor);
        set_tensor(decoder->n_conv, tensor);
    } else if (name == "n_conv_bias") {
        decoder->n_conv_bias = ggml_dup_tensor(ctx, ggml_transpose(ctx, tensor));
        set_tensor(decoder->n_conv_bias, tensor);
    } else if (name == "asr_conv_weight") {
        tensor            = squeeze_3d_2d_e0(ctx, tensor);
        decoder->asr_conv = ggml_dup_tensor(ctx, tensor);
        set_tensor(decoder->asr_conv, tensor);
    } else if (name == "asr_conv_bias") {
        decoder->asr_conv_bias = ggml_dup_tensor(ctx, ggml_transpose(ctx, tensor));
        set_tensor(decoder->asr_conv_bias, tensor);
    } else if (has_prefix(name, "decoder_blocks")) {
        std::vector<std::string> parts = split(name, ".");
        int                      i     = std::stoi(parts[1]);
        assign_ada_res_block(decoder->decoder_blocks[i], parts[2], tensor);
    } else if (has_prefix(name, "encoder_block")) {
        std::vector<std::string> parts = split(name, ".");
        assign_ada_res_block(decoder->encoder_block, parts[1], tensor);
    } else if (has_prefix(name, "generator")) {
        assign_generator_weight(decoder->generator, name.substr(10), tensor);
    }
}
