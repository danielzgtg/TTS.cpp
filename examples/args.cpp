#include "args.h"

#include <iostream>
#include <sstream>

#include "args_config.h"

void arg::print_help() const {
    cout << "--" << full_name;
    if (*abbreviation) {
        cout << " (-" << abbreviation << ")";
    }
    if (*description) {
        cout << (required ? ":\n    (REQUIRED) " : ":\n    (OPTIONAL) ") << description << ".\n";
    } else {
        cout << (required ? " is a required parameter.\n" : " is an optional parameter.\n");
    }
}

void arg::parse(span<str> & argv) {
    required = false;
    if (const auto bool_param{ get_if<bool>(&value) }) {
        *bool_param = true;
        return;
    }
    if (argv.empty()) {
        fprintf(stderr, "The option '--%s' requires an argument\n", full_name);
        exit(1);
    }
    const str a = argv[0];
    argv        = argv.subspan(1);
    if (const auto string_param{ get_if<str>(&value) }) {
        *string_param = a;
    } else if (const auto int_param{ get_if<int>(&value) }) {
        istringstream{ a } >> *int_param;
    } else if (const auto float_param{ get_if<float>(&value) }) {
        istringstream{ a } >> *float_param;
    }
}

void arg_list::parse(int argc, str argv_[]) {
    TTS_ASSERT(argc);
    span<str> argv{ argv_, static_cast<size_t>(argc) };
    argv = argv.subspan(1);
    while (!argv.empty()) {
        str name{ argv[0] };
        if (*name != '-') {
            fprintf(stderr, "Only named arguments are supported\n");
            exit(1);
        }
        ++name;
        const map<sv, size_t> * lookup = &abbreviations;
        if (*name == '-') {
            ++name;
            lookup = &full_names;
            if (name == "help"sv) {
                for (const size_t i : full_names | views::values) {
                    args[i].print_help();
                }
                exit(0);
            }
        }
        const auto found = lookup->find(sv{ name });
        if (found == lookup->end()) {
            fprintf(stderr,
                    "argument '%s' is not a valid argument. "
                    "Call '--help' for information on all valid arguments.\n",
                    argv[0]);
            exit(1);
        }
        argv = argv.subspan(1);
        args[found->second].parse(argv);
    }
    for (const arg & x : args) {
        if (x.required) {
            fprintf(stderr, "argument '--%s' is required.\n", x.full_name);
            exit(1);
        }
    }
}

void add_baseline_args(arg_list & args) {
    // runner_from_file
    args.add({ "", "model-path", "mp", "The local path of the gguf model(s) to load", true });
    args.add({ max(static_cast<int>(thread::hardware_concurrency()), 1), "n-threads", "nt",
               "The number of CPU threads to run calculations with. Defaults to known hardware concurrency. "
               "If hardware concurrency cannot be determined then it defaults to 1" });
    args.add({ false, "no-mmap", "",
               "Disable memory mapping to reduce page evictions in rare cases. "
               "This will usually slow down loading and prevent loading larger-than-memory modesl." });
}

void add_espeak_voice_arg(arg_list & args) {
    static constexpr const char * default_espeak_voice_id{ generation_configuration{}.espeak_voice_id };
    args.add({ default_espeak_voice_id, "espeak-voice-id", "eid",
               "The eSpeak voice id to use for phonemization. "
               "This should only be specified when the correct eSpeak voice cannot be inferred from the Kokoro voice. "
               "See MultiLanguage Configuration in the README for more info" });
}
