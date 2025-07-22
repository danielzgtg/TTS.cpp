
void kokoro_model::post_load_assign() {
    size_t original_offset   = offset;
    n_kernels_tensor         = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    n_kernels_tensor->buffer = buf;
    n_kernels_tensor->data   = (void *) ((uint8_t *) ggml_backend_buffer_get_base(buf) + offset);
    size_t size              = ggml_nbytes(n_kernels_tensor);
    float  nker              = (float) n_kernels;
    ggml_backend_tensor_set(n_kernels_tensor, &nker, 0, size);
    offset += size;

    sqrt_tensor         = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    sqrt_tensor->buffer = buf;
    sqrt_tensor->data   = (void *) ((uint8_t *) ggml_backend_buffer_get_base(buf) + offset);
    size                = ggml_nbytes(sqrt_tensor);
    float sqrt2         = sqrtf(2.0f);
    ggml_backend_tensor_set(sqrt_tensor, &sqrt2, 0, size);
    offset += size;

    std::vector<float> data{};
    for (int l = 0; l < lstms.size(); l++) {
        lstm *        rnn         = lstms[l];
        const int32_t hidden_size = rnn->cells[0]->biases[0]->ne[0];
        data.resize(hidden_size);

        for (int i = 0; i < rnn->cells.size(); i++) {
            ggml_tensor * h = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden_size);
            ggml_tensor * s = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden_size);
            h->buffer       = buf;
            h->data         = (void *) ((uint8_t *) ggml_backend_buffer_get_base(buf) + offset);
            size_t size     = ggml_nbytes(h);
            ggml_backend_tensor_set(h, data.data(), 0, size);
            ggml_format_name(h, "lstm%d_hidden", l);
            offset += size;
            s->buffer = buf;
            s->data   = (void *) ((uint8_t *) ggml_backend_buffer_get_base(buf) + offset);
            ggml_backend_tensor_set(s, data.data(), 0, size);
            ggml_format_name(h, "lstm%d_state", l);
            offset += size;
            rnn->hidden.push_back(h);
            rnn->states.push_back(s);
        }
        data.clear();
    }

    if (window == "hann") {
        std::vector<float> wdata;
        wdata.reserve(true_n_fft);
        hann_window(true_n_fft, wdata);
        decoder->generator->window         = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, true_n_fft);
        decoder->generator->window->buffer = buf;
        decoder->generator->window->data   = (void *) ((uint8_t *) ggml_backend_buffer_get_base(buf) + offset);
        size_t size                        = ggml_nbytes(decoder->generator->window);
        ggml_backend_tensor_set(decoder->generator->window, wdata.data(), 0, size);
        ggml_set_name(decoder->generator->window, "stft_window");
        offset += size;
        wdata.clear();
    } else {
        TTS_ABORT("Window of type %s is not supported.", window.c_str());
    }

    harmonic_sampling_norm         = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, harmonic_num + 1);
    harmonic_sampling_norm->buffer = buf;
    harmonic_sampling_norm->data   = (void *) ((uint8_t *) ggml_backend_buffer_get_base(buf) + offset);
    std::vector<float> hdata;
    hdata.reserve(harmonic_num + 1);
    for (int i = 0; i < harmonic_num + 1; i++) {
        hdata.push_back(((float) i + 1.0f) / sample_rate);
    }
    size_t hsize = ggml_nbytes(harmonic_sampling_norm);
    ggml_backend_tensor_set(harmonic_sampling_norm, hdata.data(), 0, hsize);
    hdata.clear();
    offset += hsize;

    sampling_factor_scalar         = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    sampling_factor_scalar->buffer = buf;
    sampling_factor_scalar->data   = (void *) ((uint8_t *) ggml_backend_buffer_get_base(buf) + offset);
    size_t scsize                  = ggml_nbytes(sampling_factor_scalar);
    // while it might appear that the upsampling_rate could be used here, the interpolation rate (i.e. the upsampling scale) is actually independent in the kokoro model implementation.
    float  sample_scalar           = upsample_scale * 2.0f * M_PI;
    ggml_backend_tensor_set(sampling_factor_scalar, &sample_scalar, 0, scsize);
    offset += scsize;
    post_load_tensor_bytes = 300 + offset - original_offset;
}

void kokoro_model::assign_weight(std::string name, ggml_tensor * tensor) {
    // all kokoro tensors are prepended by "kokoro" so lets trim that off and assign based on the module
    std::vector<std::string> parts = split(name, ".");
    if (parts.size() < 2) {
        return;  // handle the null context tensor;
    }
    if (parts[1] == "albert") {
        assign_albert_weight(name.substr(7 + parts[1].size() + 1), tensor);
    } else if (parts[1] == "duration_predictor") {
        assign_duration_weight(name.substr(7 + parts[1].size() + 1), tensor);
    } else if (parts[1] == "text_encoder") {
        assign_text_encoder_weight(name.substr(7 + parts[1].size() + 1), tensor);
    } else if (parts[1] == "decoder") {
        assign_decoder_weight(name.substr(7 + parts[1].size() + 1), tensor);
    } else if (parts[1] == "voice_tensors") {
        voices[parts[2]] = ggml_dup_tensor(ctx, tensor);
        set_tensor(voices[parts[2]], tensor);
    }
}

void kokoro_model::assign_text_encoder_weight(std::string name, ggml_tensor * tensor) {
    if (name == "embedding_weight") {
        text_encoder->embd = ggml_dup_tensor(ctx, tensor);
        set_tensor(text_encoder->embd, tensor);
    } else if (has_prefix(name, "lstm")) {
        assign_lstm(text_encoder->out_lstm, name.substr(5), tensor);
    } else if (has_prefix(name, "layers")) {
        std::vector<std::string> parts = split(name, ".");
        int                      i     = std::stoi(parts[1]);
        if (parts[2] == "gamma") {
            text_encoder->conv_layers[i]->norm_gamma = ggml_dup_tensor(ctx, tensor);
            set_tensor(text_encoder->conv_layers[i]->norm_gamma, tensor);
        } else if (parts[2] == "beta") {
            text_encoder->conv_layers[i]->norm_beta = ggml_dup_tensor(ctx, tensor);
            set_tensor(text_encoder->conv_layers[i]->norm_beta, tensor);
        } else if (parts[2] == "weight") {
            text_encoder->conv_layers[i]->conv_weight = ggml_dup_tensor(ctx, tensor);
            set_tensor(text_encoder->conv_layers[i]->conv_weight, tensor);
        } else if (parts[2] == "bias") {
            text_encoder->conv_layers[i]->conv_bias = ggml_dup_tensor(ctx, ggml_transpose(ctx, tensor));
            set_tensor(text_encoder->conv_layers[i]->conv_bias, tensor);
        }
    }
}
