#include "buffers.h"

// TODO finish adding GPU support here

#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "imports.h"

tts_backend_memory::tts_backend_memory(bool cpu_only) :
    cpu{ ggml_backend_cpu_init() },
    _gpu{ cpu_only ? nullptr : ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr) } {
    ggml_backend_cpu_set_n_threads(&*cpu, n_threads);
    ggml_backend_cpu_set_threadpool(&*cpu, &*threadpool);
    if (!cpu_only) {
        if (!_gpu) {
            TTS_ABORT(
                "'GGML_USE_METAL' is not defined either set the model to use CPU only or install ggml with metal "
                "support.");
        }
#ifdef GGML_USE_METAL
        // this is form copied from llama.cpp, but has since been removed. I don't know if this should be tuned.
        ggml_backend_metal_set_n_cb(backend, 1);
#endif
    }
}

tts_backend_memory::~tts_backend_memory() {
    ggml_backend_free(cpu);
    ggml_backend_free(_gpu);
}

tts_buffer::tts_buffer(int n_threads) :
    threadpool{ [n_threads] {
        ggml_threadpool_params ttp{ ggml_threadpool_params_default(n_threads) };
        return ggml_threadpool_new(&ttp);
    }() } {

}

tts_buffer::~tts_buffer() {
    ggml_free(ctx);
    ggml_backend_buffer_free(buf);
    ggml_threadpool_free(threadpool);
}

void tts_buffer::set_tensor(ggml_tensor * tensor, ggml_tensor * target) {
    tensor->buffer    = buf;
    tensor->data      = static_cast<uint8_t *>(ggml_backend_buffer_get_base(buf)) + offset;
    const size_t size = ggml_nbytes(target);
    ggml_backend_tensor_set(tensor, target->data, 0, size);
    ggml_set_name(tensor, target->name);
    offset += size;
}
