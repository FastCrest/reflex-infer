#include "reflex/infer.h"

#include <atomic>

namespace reflex::infer {
namespace {

std::atomic<GemvQuantFn> g_gemv_quant{nullptr};
std::atomic<GemvQuantAddFn> g_gemv_quant_add{nullptr};
std::atomic<GemvQuantPairFn> g_gemv_quant_pair{nullptr};
std::atomic<GemvQuantTripleFn> g_gemv_quant_triple{nullptr};
std::atomic<GemmQuantBatchedFn> g_gemm_quant_batched{nullptr};
std::atomic<GemvQuantF32Fn> g_gemv_quant_f32{nullptr};

template <typename Fn>
Fn load_backend(const std::atomic<Fn>& slot) noexcept {
    return slot.load(std::memory_order_acquire);
}

template <typename Fn>
void store_backend(std::atomic<Fn>& slot, Fn fn) noexcept {
    slot.store(fn, std::memory_order_release);
}

}  // namespace

void register_q4_backend(const Q4Kernels& kernels) noexcept {
    store_backend(g_gemv_quant, kernels.gemv_quant);
    store_backend(g_gemv_quant_add, kernels.gemv_quant_add);
    store_backend(g_gemv_quant_pair, kernels.gemv_quant_pair);
    store_backend(g_gemv_quant_triple, kernels.gemv_quant_triple);
    store_backend(g_gemm_quant_batched, kernels.gemm_quant_batched);
    store_backend(g_gemv_quant_f32, kernels.gemv_quant_f32);
}

Q4Kernels registered_q4_backend() noexcept {
    Q4Kernels kernels{};
    kernels.gemv_quant = load_backend(g_gemv_quant);
    kernels.gemv_quant_add = load_backend(g_gemv_quant_add);
    kernels.gemv_quant_pair = load_backend(g_gemv_quant_pair);
    kernels.gemv_quant_triple = load_backend(g_gemv_quant_triple);
    kernels.gemm_quant_batched = load_backend(g_gemm_quant_batched);
    kernels.gemv_quant_f32 = load_backend(g_gemv_quant_f32);
    return kernels;
}

Status gemv_quant(const GemvQuantArgs& args) {
    if (auto fn = load_backend(g_gemv_quant)) {
        return fn(args);
    }
    return Status::NotImplemented;
}

Status gemv_quant_add(const GemvQuantAddArgs& args) {
    if (auto fn = load_backend(g_gemv_quant_add)) {
        return fn(args);
    }
    return Status::NotImplemented;
}

Status gemv_quant_pair(const GemvQuantPairArgs& args) {
    if (auto fn = load_backend(g_gemv_quant_pair)) {
        return fn(args);
    }
    return Status::NotImplemented;
}

Status gemv_quant_triple(const GemvQuantTripleArgs& args) {
    if (auto fn = load_backend(g_gemv_quant_triple)) {
        return fn(args);
    }
    return Status::NotImplemented;
}

Status gemm_quant_batched(const GemmQuantBatchedArgs& args) {
    if (auto fn = load_backend(g_gemm_quant_batched)) {
        return fn(args);
    }
    return Status::NotImplemented;
}

Status gemv_quant_f32(const GemvQuantF32Args& args) {
    if (auto fn = load_backend(g_gemv_quant_f32)) {
        return fn(args);
    }
    return Status::NotImplemented;
}

}  // namespace reflex::infer
