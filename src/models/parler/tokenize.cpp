
static parler_ubatch batch_from_sentence(std::string sentence, parler_tts_model * model, unigram_tokenizer * tokenizer) {
    parler_ubatch batch;
    batch.audio_generation = false;
    std::vector<uint32_t>* token_ids = new std::vector<uint32_t>;
    tokenizer->tokenize(sentence, *token_ids);
    token_ids->push_back(tokenizer->eos_token);
    batch.current_step = 0;
    batch.n_tokens = token_ids->size();
    batch.n_audio_tokens = 0;
    batch.sequence_length = batch.n_tokens; // sequence_length is equal to the number of tokens for non-audio generation
    std::vector<uint32_t>* position = new std::vector<uint32_t>;
    for (uint32_t i = 0; i < batch.sequence_length; i++) {
        position->push_back(i);
    }
    std::vector<uint32_t>* order = new std::vector<uint32_t>;
    for (int i = 0; i < batch.sequence_length; i++) {
        if (i >= batch.sequence_length - 1) {
            order->push_back(0);
        } else {
            order->push_back(i+1);
        }
    }
    batch.positions = position->data();
    batch.tokens = token_ids->data();
    return batch;
}
