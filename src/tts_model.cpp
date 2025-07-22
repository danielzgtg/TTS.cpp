#include "tts_model.h"

#include "ggml-backend.h"
#include "ggml-cpu.h"

void append_to_response(tts_response * response, const tts_response * to_append) {
    auto * new_data = static_cast<float *>(malloc((response->n_outputs + to_append->n_outputs) * sizeof(float)));
    if (response->n_outputs > 0) {
        std::memcpy(new_data, response->data, response->n_outputs * sizeof(float));
    }
    if (to_append->n_outputs > 0) {
        float * next_loc = new_data + response->n_outputs;
        std::memcpy(next_loc, to_append->data, to_append->n_outputs * sizeof(float));
    }
    response->data = new_data;
    response->n_outputs += to_append->n_outputs;
}

/* 
 * Pulls output_size to prepped buffer 'output' from 'output_node' tensor. If no buffer is passed will default to the existing output buffer present 
 * on runner_context. 
 */
void runner_context::get_ggml_node_data(ggml_tensor * output_node, float * output, size_t output_size,
                                        ggml_backend_buffer_t buffer) const {
    if (buffer == nullptr) {
        buffer = buf_output;
    }
    if (ggml_backend_buffer_get_size(buffer) < output_size) {
        TTS_ABORT("Output buffer overflow of %d / %d for output node '%s'\n", output_size,
                  ggml_backend_buffer_get_size(buffer), ggml_get_name(output_node));
    } if (ggml_nbytes(output_node) < output_size) {
        TTS_ABORT("Output node, '%s', with %d bytes is too small for #ggml_backend_tensor_get_async with size of %d.\n",
                  ggml_get_name(output_node), ggml_nbytes(output_node), output_size);
    }
    const ggml_backend_t backend_res = ggml_backend_sched_get_tensor_backend(sched, output_node);
    ggml_backend_tensor_get_async(backend_res, output_node, output, 0, output_size);
}

void runner_context::set_threads() {
    if (backend != nullptr) {
#ifdef GGML_USE_METAL
        // this is form copied from llama.cpp, but has since been removed. I don't know if this should be tuned.
        ggml_backend_metal_set_n_cb(backend, 1);
#endif
    }
    if (backend_cpu != nullptr) {
        ggml_backend_cpu_set_n_threads(backend_cpu, n_threads);
        ggml_threadpool_params ttp = ggml_threadpool_params_default(n_threads);
        threadpool                 = ggml_threadpool_new(&ttp);
        ggml_backend_cpu_set_threadpool(backend_cpu, threadpool);
    }
}

void runner_context::build_schedule(size_t max_nodes) {
    backend_cpu_buffer = ggml_backend_cpu_buffer_type();
    if (backend != nullptr) {
#ifdef GGML_USE_METAL
        backend_buffer = ggml_backend_metal_buffer_type();
#endif
        vector<ggml_backend_buffer_type_t> bufs  = { backend_buffer, backend_cpu_buffer };
        vector<ggml_backend_t>             backs = { backend, backend_cpu };
        sched = ggml_backend_sched_new(backs.data(), bufs.data(), 2, max_nodes, false);
    } else {
        vector<ggml_backend_buffer_type_t> bufs  = { backend_cpu_buffer };
        vector<ggml_backend_t>             backs = { backend_cpu };
        sched = ggml_backend_sched_new(backs.data(), bufs.data(), 1, max_nodes, false);
    }
}

bool runner_context::prep_schedule(ggml_cgraph * gf) const {
    return ggml_backend_sched_reserve(sched, gf);
}

tts_runner::tts_runner(const tts_model_loader & loader) : loader{ ref(loader) } {}

tts_runner::~tts_runner() = default;

void tts_runner::init_build(std::vector<uint8_t> * buf_compute_meta) {
    ctx = ggml_init({
        .mem_size{ buf_compute_meta->size() },
        .mem_buffer{ buf_compute_meta->data() },
        .no_alloc{ true },
    });
}

void tts_runner::free_build() {
    ggml_free(ctx);
    ctx = nullptr;
}

void tts_model::prep_buffers_and_context(bool cpu_only, float size_offset, uint32_t dedicated_add_on_size) {
    backends.ctx = ggml_init({
        .mem_size{ static_cast<const size_t>(ggml_tensor_overhead() * (tensor_meta.n_tensors * size_offset)) },
        .mem_buffer{ nullptr },
        .no_alloc{ true },
    });
    // currently DAC is only supported on cpu because the ops are not implemented on other devices;
    if (cpu_only) {
        backend = ggml_backend_cpu_init();
        buffer  = ggml_backend_cpu_buffer_type();
    } else {
#ifdef GGML_USE_METAL
        backend = ggml_backend_metal_init();
        buffer  = ggml_backend_metal_buffer_type();
#endif
        // if use metal is not installed then we need to warn here
        if (!backend || !buffer) {
            TTS_ABORT(
                "'GGML_USE_METAL' is not defined either set the model to use CPU only or install ggml with metal "
                "support.");
        }
    }
    backends.buf = ggml_backend_buft_alloc_buffer(buffer, tensor_meta.n_bytes + dedicated_add_on_size);
}

void tts_model::setup_from_file(ggml_context * load_context, bool cpu_only, const string & model_prefix,
                                float size_offset, uint32_t dedicated_add_on_size) {
    tensor_meta = compute_tensor_meta(model_prefix, load_context, compute_tensor_meta_cb);
    prep_buffers_and_context(cpu_only, size_offset, dedicated_add_on_size);
}

size_t tts_model::max_nodes() const {
    return max<size_t>(8192, tensor_meta.n_tensors * 5);
}

tts_model::~tts_model() {
    ggml_backend_free(backend);
}
