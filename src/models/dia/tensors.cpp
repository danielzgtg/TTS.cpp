
void dia_model::assign_weight(string name, ggml_tensor * tensor) {
    vector<string> parts = split(name, ".");
    TTS_ASSERT(parts.size() >= 3);

    if (parts[1] == "encoder") {
        assign_to_encoder(parts, tensor, name);
    } else if (parts[1] == "decoder") {
        assign_to_decoder(parts, tensor, name);
    } else {
        TTS_ABORT("Unrecognized tensor '%s' when loading Dia from GGUF file.", name.c_str());
    }
}

void dia_runner::assign_weight(string name, ggml_tensor * tensor) {
    if (tensor->data == NULL) {
        return;
    }

    if (name.size() == 0) {
        // handles the top level meta tensor
        return;
    }

    if (name.size() > 14 && name.substr(0, 14) == "audio_encoder.") {
        dac_runner->model->assign_weight(name.substr(14), tensor);
    } else {
        model->assign_weight(name, tensor);
    }
}
