
void dia_runner::tokenize_sentence(string sentence, dia_ubatch & batch) {
    // Dia's tokenization process is unusual.
    // Essentially Dia takes the byte value for each character and uses that as a token array.
    // Additionally, because Dia performs a cfg-scale adjustment before sampling tokens, it is necessary to
    // generate with a conditioned context (i.e. with the text)
    // and an unconditioned context (i.e. without any text) so that
    // proper adjustments can be perfored at each generation step.
    // This means that we need to pad the end of our tokens to the
    // max context size for both the conditional and unconditional sequence.

    // if the sentence isn't prepended by dialogue start tokens, [S1] or [S2], then append one.
    sentence     = strip(sentence);
    string start = sentence.substr(0, 4);
    if (start != "[S1]" && start != "[S2]") {
        sentence = "[S1] " + sentence;
    }
    if (sentence[sentence.size() - 1] != '.') {
        sentence += ".";
    }

    // [S1] and [S2] are special character sequences that are replaced with the special tokens 0x01 and 0x02 respectively.
    string r1(1, 1);
    string r2(1, 2);
    while (sentence.find("[S1]") != string::npos) {
        size_t pos = sentence.find("[S1]");
        sentence.replace(pos, 4, r1);
    }
    while (sentence.find("[S2]") != string::npos) {
        size_t pos = sentence.find("[S2]");
        sentence.replace(pos, 4, r2);
    }

    if (sentence.size() > model->max_encoder_context_length) {
        TTS_ABORT("Dia currently only supports a max of %d characters and received an input of %d characters.",
                  model->max_encoder_context_length, sentence.size());
    }
    batch.tokens.reserve(model->max_encoder_context_length * 2);
    for (auto character : sentence) {
        batch.tokens.push_back((uint32_t) character);
    }
    batch.sentence_length = batch.tokens.size();
    // this 100 token warning is arbitrarily chosen based on spot checking small prompt performance
    if (batch.sentence_length <= 100) {
        fprintf(stdout,
                "Your prompt has fewer than 100 tokens. Please note that Dia's generation with prompts that are fewer "
                "than 100 tokens is highly inconsistent.\n");
    }

    for (int i = (int) batch.tokens.size(); i < model->max_encoder_context_length * 2; i++) {
        batch.tokens.push_back(0u);
    }
}

dia_ubatch dia_runner::batch_from_sentence(string sentence) {
    // if we are generating a new batch from tokens then we need to run the encoder step;
    dia_ubatch batch{ 1, true };
    tokenize_sentence(sentence, batch);
    batch.audio_tokens.reserve(model->n_output_heads);
    for (int i = 0; i < model->n_output_heads; i++) {
        batch.audio_tokens.push_back(model->bos_token_id);
    }
    return batch;
}
