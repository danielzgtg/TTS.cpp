#include "quantize_impl.h"

#if defined(__unix__) || defined(__APPLE__)
#    include <fcntl.h>
#endif

#include <cstring>
#include <filesystem>
#include <fstream>
#include <thread>

#include "ggml-cpp.h"
#include "ggml-iterator.h"
#include "llama-mmap.h"

// Reuse some fields that are only used during inferencing
#define out_tensor_nb view_offs
#define out_job_index padding
#define out_data      extra

template <bool plan_only> static void quantize_tensor(ggml_type qtype, ggml_tensor * tensor, bool single_thread) {
    int chunk_size_multiplier{ 1 };
    if (qtype == GGML_TYPE_Q4_0_4_4 || qtype == GGML_TYPE_Q4_0_4_8 || qtype == GGML_TYPE_Q4_0_8_8) {
        if (qtype == GGML_TYPE_Q4_0_8_8 && tensor->ne[1] % 8 != 0 || tensor->ne[1] % 4 != 0) {
            qtype = GGML_TYPE_Q4_0;
        }
        if (qtype == GGML_TYPE_Q4_0_8_8) {
            chunk_size_multiplier = 8;
        } else if (qtype == GGML_TYPE_Q4_0_4_4 || qtype == GGML_TYPE_Q4_0_4_8) {
            chunk_size_multiplier = 4;
        }
    }
    const int64_t d3_step{ tensor->ne[0] * tensor->ne[1] };
    const int64_t n_per_row{ tensor->ne[0] };
    const int64_t nrows{ tensor->ne[1] };
    const size_t  row_size{ ggml_row_size(qtype, n_per_row) };

    static constexpr int64_t min_chunk_size{ 32 * 512 };
    const int64_t            chunk_size{ chunk_size_multiplier * n_per_row *
                              (n_per_row >= min_chunk_size ? 1 : (min_chunk_size + n_per_row - 1) / n_per_row) };
    const int64_t            nrows_per_chunk{ single_thread ? nrows : chunk_size / n_per_row };

    if constexpr (plan_only) {
        tensor->out_tensor_nb = 0;
    }
    const int64_t nchunks_per_d3{ (nrows + nrows_per_chunk - 1) / nrows_per_chunk };
    const int64_t njobs{ nchunks_per_d3 * tensor->ne[2] };
    TTS_ASSERT(njobs);
    atomic<int64_t> & job_index_{ reinterpret_cast<atomic<int64_t> &>(tensor->out_job_index) };
    int64_t           job_index;
    while ((job_index = job_index_.fetch_add(1, memory_order_relaxed)) < njobs) {
        const int64_t       d3_index{ job_index / nchunks_per_d3 };
        const float * const f32_data_d3 = static_cast<float *>(tensor->data) + d3_index * d3_step;
        char * const        new_data_d3 = static_cast<char *>(tensor->out_data) + d3_index * row_size * nrows;

        const int64_t i1{ job_index % nchunks_per_d3 * nrows_per_chunk };
        const int64_t this_nrow = min(nrows - i1, nrows_per_chunk);
        const size_t  plan_size{ this_nrow * row_size };
        if constexpr (plan_only) {
            tensor->out_tensor_nb += plan_size;
            continue;
        }

        const size_t this_size{ ggml_quantize_chunk(qtype, f32_data_d3, new_data_d3, i1 * n_per_row, this_nrow,
                                                    n_per_row, nullptr) };
        TTS_ASSERT(plan_size == this_size);

        if (!single_thread) {
            // validate the quantized data; I am not sure how this would occur,
            // but there is always the safe fallback on doing this single threaded.
            TTS_ASSERT(ggml_validate_row_data(qtype, new_data_d3 + i1 * row_size, plan_size));
        }
    }
    if constexpr (plan_only) {
        tensor->out_tensor_nb = GGML_PAD(tensor->out_tensor_nb, GGUF_DEFAULT_ALIGNMENT);
    }
}

static void quantization_worker(const ggml_context * weight_ctx, const tts_model_quantizer & quantizer,
                                const quantization_params & params, bool single_thread) {
    for (ggml_tensor & cur : ggml_tensor_iterator{ *weight_ctx }) {
        if (!*cur.name) {
            continue;
        }
        atomic<int64_t> & job_index_{ reinterpret_cast<atomic<int64_t> &>(cur.out_job_index) };
        if (const ggml_type wanted_type{ quantizer.get_quantize_type(cur.name, params) }) {
            quantize_tensor<false>(wanted_type, &cur, single_thread);
        } else if (!job_index_.fetch_add(1, memory_order_relaxed)) {
            memcpy(cur.out_data, cur.data, cur.out_tensor_nb);
        }
    }
}

void quantize_gguf(str ifile, str ofile, const quantization_params & params) {
    switch (params.quantize_type) {
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_F16:
        case GGML_TYPE_Q4_0:
            break;
        default:
            fprintf(stdout, "Warning: Untested quantization type '%d'. Use at your own risk.\n", params.quantize_type);
    }

    llama_file in_mmap_file{ ifile, "rb" };
    // create/truncate/zero file and assume nobody else will touch it
    thread     fallocate_thread{ [ofile, fallocate_size = in_mmap_file.size()] {
#if defined(__unix__) || defined(__APPLE__)
        const int fd = creat(ofile, 0666);
        static_cast<void>(!ftruncate(fd, fallocate_size));
        posix_fallocate(fd, 0, fallocate_size);
        close(fd);
#else
        static_cast<void>(ofstream{ ofile });
        filesystem::resize_file(ofile, fallocate_size);
#endif
    } };

    unique_ptr<llama_mmap> in_mmap{};
    void *                 in_buffer{};
    if (llama_mmap::SUPPORTED) {
        in_mmap   = make_unique<llama_mmap>(&in_mmap_file);
        in_buffer = in_mmap->addr();
    }
    ggml_context *         weight_ctx{};
    const gguf_context_ptr in_gguf{ gguf_init_from_file(ifile, {
                                                                   .no_alloc{ llama_mmap::SUPPORTED },
                                                                   .ctx{ &weight_ctx },
                                                               }) };
    TTS_ASSERT(in_gguf);
    TTS_ASSERT(gguf_find_key(&*in_gguf, "general.alignment") == -1);
    const tts_model_quantizer & quantizer{ quantizer_from_gguf(&*in_gguf) };
    if (llama_mmap::SUPPORTED) {
        in_buffer = static_cast<char *>(in_buffer) + gguf_get_data_offset(&*in_gguf);
        const int n{ gguf_get_n_tensors(&*in_gguf) };
        int       i{};
        for (ggml_tensor & cur : ggml_tensor_iterator{ *weight_ctx }) {
            TTS_ASSERT(i < n);
            TTS_ASSERT(!strcmp(cur.name, gguf_get_tensor_name(&*in_gguf, i)));
            cur.data = static_cast</*const*/ char *>(in_buffer) + gguf_get_tensor_offset(&*in_gguf, i);
            ++i;
        }
    }

    const gguf_context_ptr out_gguf{ gguf_init_empty() };
    // copy the KV pairs from the input file
    gguf_set_kv(out_gguf.get(), &*in_gguf);
    gguf_set_val_u32(out_gguf.get(), "general.quantization_version", GGML_QNT_VERSION);
    gguf_set_val_u32(out_gguf.get(), "general.quantization_type", params.quantize_type);

    const bool single_thread{ params.n_threads <= 1 };
    size_t     out_tensors_nb{};
    // Pass 1/3: Calculate sizes
    for (ggml_tensor & cur : ggml_tensor_iterator{ *weight_ctx }) {
        if (!*cur.name) {
            continue;
        }
        gguf_add_tensor(&*out_gguf, &cur);
        if (const ggml_type wanted_type{ quantizer.get_quantize_type(cur.name, params) }) {
            if (cur.type != GGML_TYPE_F32) {
                TTS_ABORT("Quantization must be from F32. Tensor '%s' has improper type '%d'\n", cur.name, cur.type);
            }
            if (wanted_type >= GGML_TYPE_IQ2_XXS && wanted_type <= GGML_TYPE_IQ4_XS) {
                TTS_ABORT("Quantization type '%d' requires an importance matrix.\n", wanted_type);
            }
            quantize_tensor<true>(wanted_type, &cur, single_thread);
            gguf_set_tensor_type(out_gguf.get(), cur.name, wanted_type);
            gguf_set_tensor_data(out_gguf.get(), cur.name, nullptr, cur.out_tensor_nb);
        } else {
            cur.out_tensor_nb = ggml_nbytes(&cur);
        }
        out_tensors_nb = GGML_PAD(out_tensors_nb + cur.out_tensor_nb, GGUF_DEFAULT_ALIGNMENT);
        fprintf(stdout, "Planning tensor: '%s' with new size: %zu bytes\n", cur.name, cur.out_tensor_nb);
    }
    const size_t out_meta_nb{ gguf_get_meta_size(out_gguf.get()) };
    const size_t out_nb{ out_meta_nb + out_tensors_nb };

    TTS_ASSERT(out_nb < in_mmap_file.size());
    if (fallocate_thread.joinable()) {
        fallocate_thread.join();
    }
    filesystem::resize_file(ofile, out_nb);
    llama_file             out_mmap_file{ ofile, "rb+" };
    unique_ptr<llama_mmap> out_mmap{};
    char *                 out_buffer{};
    if (llama_mmap::SUPPORTED) {
        out_mmap   = make_unique<llama_mmap>(&out_mmap_file, -1, false, true);
        out_buffer = static_cast<char *>(out_mmap->addr());
    } else {
        out_buffer = new char[out_nb]{};
    }

    gguf_get_meta_data(out_gguf.get(), out_buffer);
    out_tensors_nb = out_meta_nb;
    TTS_ASSERT(out_tensors_nb == GGML_PAD(out_tensors_nb, GGUF_DEFAULT_ALIGNMENT));
    // Pass 2/3 Initialize output pointers
    for (ggml_tensor & cur : ggml_tensor_iterator{ *weight_ctx }) {
        if (!*cur.name) {
            continue;
        }
        new (cur.out_job_index) atomic_int64_t{};
        cur.out_data   = out_buffer + out_tensors_nb;
        out_tensors_nb = GGML_PAD(out_tensors_nb + cur.out_tensor_nb, GGUF_DEFAULT_ALIGNMENT);
    }
    // Pass 3/3 Quantize in parallel
    if (params.n_threads > 1) {
        vector<thread> threads{};
        threads.reserve(params.n_threads);
        for (int i = 1; i < params.n_threads; i++) {
            threads.emplace_back(&quantization_worker, weight_ctx, cref(quantizer), cref(params), single_thread);
        }
        quantization_worker(weight_ctx, quantizer, params, single_thread);
        for (auto & t : threads) {
            t.join();
        }
    } else {
        quantization_worker(weight_ctx, quantizer, params, single_thread);
    }

    if (!out_mmap) {
        out_mmap_file.write_raw(out_buffer, out_nb);
        delete[] out_buffer;
    }
}
