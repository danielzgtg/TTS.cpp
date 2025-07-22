#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <memory>
#include <ranges>
#include <string_view>
#include <vector>

using namespace std;
using namespace std::string_view_literals;
typedef std::string_view sv;
typedef const char *     str;

#define GGML_ABORT(...) ggml_abort(__FILE__, __LINE__, __VA_ARGS__)
extern "C" {
#ifdef GGML_SHARED
#    if defined(_WIN32) && !defined(__MINGW32__)
__declspec(dllimport)
#    endif
#endif
[[noreturn]] __attribute__((__format__(printf, 3, 4))) extern void
ggml_abort(const char * file, int line, const char * fmt, ...);
}
#define TTS_ABORT GGML_ABORT
#define TTS_ASSERT(x) \
    if (!(x))         \
    TTS_ABORT("TTS_ASSERT(%s) failed", #x)

struct ggml_context;
struct ggml_tensor;
