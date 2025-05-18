#include "tokenizer.h"

void token_trie::add(const string & gram, uint32_t token) {
    _add(gram, token, 0);
}

void token_trie::_add(const string & gram, uint32_t new_token, size_t index) {
    if (index >= gram.size()) {
        has_value = true;
        token = new_token;
        return;
    }
    const char c = gram[index];
    auto res = children.find(c);
    if (res != children.end()) {
        res->second._add(gram, new_token, index + 1);
    } else {
        struct token_trie nt{};
        nt._add(gram, new_token, index + 1);
        children[c] = nt;
    }
}

const struct token_trie * token_trie::traverse(const char c) const {
    auto res = children.find(c);
    if (res != children.end()) {
        return &res->second;
    }

    return NULL;
}

size_t unicode_len_utf8(char src) {
    const size_t lookup[] = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 4 };
    uint8_t highbits = static_cast<uint8_t>(src) >> 4;
    return lookup[highbits];
}

// the general approach here is to find the character grams that sum to the max possible value over the entire text sequence.
// The particular algorithm used here effectively works by walking the text and at each index storing the max value of all possible gram combinations
// we can then reverse that sequence to pick the best possible tokens.
void unigram_tokenizer::tokenize(const string & text, vector<uint32_t> & tokens) {
    // the parler tokenizer's normalizer (i.e. the bert normalizer implemented by huggingface tokenizers libs) only deduplicates and strips extra spaces and
    // optionally handles chinese characters and accents (neither of which are currently supported here).
    string normalized = text;
    if (dedupe_spaces) {
        normalized = " " + regex_replace(text, duped_spaces, " ");
    }
    
    size_t text_length = normalized.size();

    // initialize score_sum to neg infinity so it will be always lower than sums of token scores
    vector<struct result> results(text_length + 1, {unk_token, 0, -INFINITY});
    results[0] = { unk_token, 0, 0 };
    
    size_t offset = 0;

    while (offset < text_length) {
        size_t current_offset = offset;
        // pulled this directly from llama.cpp; I suspect that this is for handling of non-utf8 steps (to be marked as unknown tokens)
        size_t n_utf8_code_units = min<size_t>(unicode_len_utf8(normalized[offset]), text_length - offset);

        bool found_unknown = true;
        const struct result & current_best = results[offset];
        
        // find the current branch in the trie
        const struct token_trie * node = root_trie.traverse(normalized[current_offset++]);
        // search for the next token
        while (current_offset <= text_length && node != NULL) {
            // check if this is a complete token (it could just be an unkown step between two tokens).
            if (node->has_value) {
                // check if it corresponds to the whole utf8 step
                if (current_offset - offset == n_utf8_code_units) {
                    found_unknown = false;
                }
                float score = current_best.score + scores[node->token];
                struct result & current_champ = results[current_offset];
                if (score > current_champ.score) {
                    struct result challenger = { node->token, offset, score };
                    current_champ = challenger;
                }
            }
            node = node->traverse(normalized[current_offset++]);
        }

        // if we found an unknown token, process it
        if (found_unknown) {
            current_offset = offset + n_utf8_code_units;
            struct result & current_champ = results[current_offset];
            float score = current_best.score + unk_token_score;
            if (score > current_champ.score) {
                struct result challenger = { unk_token, offset, score };
                current_champ = challenger;
            }
        }

        // move one utf8 step
        offset += n_utf8_code_units;
    }

    // if we have more than on unknown token in a row, we can join them.
    bool is_prev_unknown = false;
    // iterate from the last result backwards and get the best performing tokens
    for (struct result & result = results[text_length]; ; result = results[result.offset]) {
        bool is_unknown = result.token == unk_token;
        if (!(is_prev_unknown && is_unknown)) {
            tokens.push_back(result.token);
        }
        if (result.offset == 0) {
            break;
        }
        is_prev_unknown = is_unknown;
    }

    // reverse the tokens since we added tokens starting from the end of the input
    reverse(tokens.begin(), tokens.end());
}

// loading the vocab to the tokenizer from gguf file.
unigram_tokenizer::unigram_tokenizer(gguf_context * meta) {
    vector<float> scores;
    int vocab_key = gguf_find_key(meta, "tokenizer.ggml.tokens");
    int vocab_size = gguf_get_arr_n(meta, vocab_key);
    scores.reserve(vocab_size);
    for (int i = 0; i < vocab_size; i++) {
        string val = gguf_get_arr_str(meta, vocab_key, i);
        vocab[val] = (uint32_t) i;
    }
    int scores_key = gguf_find_key(meta, "tokenizer.ggml.scores");
    int scores_size = gguf_get_arr_n(meta, scores_key);
    assert(scores_size == vocab_size);
    scores.reserve(scores_size);
    scores.append_range(span{(float*) gguf_get_arr_data(meta, scores_key), scores_size});
    int unkown_token_key = gguf_find_key(meta, "tokenizer.ggml.unknown_token_id");
    unk_token = gguf_get_val_u32(meta, unkown_token_key);
    unk_token_score = scores[unk_token];
    uint32_t eos_token_key = gguf_find_key(meta, "tokenizer.ggml.eos_token_id");
    if (eos_token_key != -1) {
        tokenizer->eos_token = gguf_get_val_u32(meta, eos_token_key);
    }
    for (const auto it : vocab) {
        root_trie.add(it.first, it.second);
    }
    return tokenizer;
}

void single_pass_tokenizer::tokenize(const string & text, vector<uint32_t> & token_ids) {
    string remaining = text;
    while (remaining.size() > 0) {
        uint32_t token_id = unknown_id;
        for (int i = 1; i < min(remaining.size()+1, max_size+1); i++) {
            string part = remaining.substr(0, i);
            ptrdiff_t pos = distance(tokens.begin(), find(tokens.begin(), tokens.end(), part));
            if (pos < tokens.size()) {
                token_id = (uint32_t) pos;
                remaining = remaining.substr(part.size(), remaining.size() - part.size());
                break;
            }
        }
        if (token_id == unknown_id) {
            remaining = remaining.substr(1, remaining.size() - 1);
        }
        token_ids.push_back(token_id);
    }
}

void single_pass_tokenizer::token_split(const string & text, vector<string> & tokens) {
    string remaining = text;
    while (remaining.size() > 0) {
        // String copying is much slower than using a string_view, but the former is simpler to implement for now.
        string token = remaining.substr(0, 1);
        for (int i = 1; i < remaining.size(); i++) {
            string part = remaining.substr(0, i+1);
            if (token_vocab.find(part) == token_vocab.end()) {
                break;
            }
            token = part;
        }
        tokens.push_back(token);
        remaining = remaining.substr(token.size(), remaining.size() - token.size());
    }
}

struct single_pass_tokenizer * single_pass_tokenizer_from_gguf(gguf_context * meta, string key_name) {
    int tokens_key = gguf_find_key(meta, key_name.c_str());
    if (tokens_key == -1) {
        TTS_ABORT("The '%s' key must be set in order to support single pass tokenization.", key_name.c_str());
    }
    vector<string> tokens;
    int token_count = gguf_get_arr_n(meta, tokens_key);
    for (int i = 0; i < token_count; i++) {
        tokens.push_back(gguf_get_arr_str(meta, tokens_key, i));
    }
    return new single_pass_tokenizer(tokens);
}

