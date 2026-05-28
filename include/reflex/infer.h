#pragma once

#include <cstddef>
#include <cstdint>

#ifndef REFLEX_INFER_VERSION
#define REFLEX_INFER_VERSION "0.1.0"
#endif

#ifndef REFLEX_INFER_HAS_CUDA
#define REFLEX_INFER_HAS_CUDA 0
#endif

namespace reflex::infer {

enum class Status : std::uint8_t {
    Success,
    UnsupportedDevice,
    UnsupportedModel,
    UnsupportedQuantFormat,
    InvalidArgument,
    NotImplemented,
};

enum class DeviceClass : std::uint8_t {
    OrinNano8GB,
    OrinNX16GB,
    AGXOrin64GB,
    Thor,
    Unknown,
};

enum class ModelFamily : std::uint8_t {
    Qwen,
    Phi,
    Llama,
    Gemma,
    Unknown,
};

enum class QuantFormat : std::uint8_t {
    Q4_0,
    Q4_K_M,
    AWQ4,
    GPTQ4,
    NF4,
    FP16,
    Unknown,
};

struct HardwareProfile {
    DeviceClass device = DeviceClass::Unknown;
    int sm = 0;
    int sm_count = 0;
    std::size_t shared_mem_per_block = 0;
    std::size_t total_memory_bytes = 0;
    float nominal_dram_gbps = 0.0f;
};

struct ModelShape {
    ModelFamily family = ModelFamily::Unknown;
    int hidden_size = 0;
    int intermediate_size = 0;
    int num_layers = 0;
    int num_heads = 0;
    int num_kv_heads = 0;
    int head_dim = 0;
    int vocab_size = 0;
    int max_context = 0;
};

struct QuantDescriptor {
    QuantFormat format = QuantFormat::Unknown;
    int block_size = 0;
    int group_size = 0;
    bool activations_fp16 = true;
    bool kv_cache_fp16 = true;
    bool kv_cache_int8 = false;
};

struct KernelSupport {
    bool q4_gemv = false;
    bool q4_mmq_prefill = false;
    bool decode_attention = false;
    bool rope = false;
    bool rmsnorm = false;
    bool kv_convert = false;
};

using StreamHandle = void*;

// `weights` is the runtime-owned logical pointer. `weights_device` is the
// CUDA-visible alias when the runtime can resolve one for mmap-backed weights.
struct GemvQuantArgs {
    void* y = nullptr;
    const void* weights = nullptr;
    const void* weights_device = nullptr;
    int ggml_type = 0;
    const void* x = nullptr;
    int M = 0;
    int K = 0;
    StreamHandle stream = nullptr;
};

struct GemvQuantAddArgs {
    void* y = nullptr;
    const void* weights = nullptr;
    const void* weights_device = nullptr;
    int ggml_type = 0;
    const void* x = nullptr;
    const void* residual = nullptr;
    int M = 0;
    int K = 0;
    StreamHandle stream = nullptr;
};

struct GemvQuantPairArgs {
    void* y0 = nullptr;
    const void* weights0 = nullptr;
    const void* weights0_device = nullptr;
    int ggml_type0 = 0;
    int M0 = 0;
    void* y1 = nullptr;
    const void* weights1 = nullptr;
    const void* weights1_device = nullptr;
    int ggml_type1 = 0;
    int M1 = 0;
    const void* x = nullptr;
    int K = 0;
    StreamHandle stream = nullptr;
};

struct GemvQuantTripleArgs {
    void* y0 = nullptr;
    const void* weights0 = nullptr;
    const void* weights0_device = nullptr;
    int ggml_type0 = 0;
    int M0 = 0;
    void* y1 = nullptr;
    const void* weights1 = nullptr;
    const void* weights1_device = nullptr;
    int ggml_type1 = 0;
    int M1 = 0;
    void* y2 = nullptr;
    const void* weights2 = nullptr;
    const void* weights2_device = nullptr;
    int ggml_type2 = 0;
    int M2 = 0;
    const void* x = nullptr;
    int K = 0;
    StreamHandle stream = nullptr;
};

struct GemmQuantBatchedArgs {
    void* y = nullptr;
    const void* weights = nullptr;
    const void* weights_device = nullptr;
    int ggml_type = 0;
    const void* x = nullptr;
    int M = 0;
    int N = 0;
    int K = 0;
    StreamHandle stream = nullptr;
};

struct GemvQuantF32Args {
    float* y = nullptr;
    const void* weights = nullptr;
    const void* weights_device = nullptr;
    int ggml_type = 0;
    const void* x = nullptr;
    int M = 0;
    int K = 0;
    StreamHandle stream = nullptr;
};

struct DequantEmbeddingRowArgs {
    void* dst = nullptr;
    const void* weights = nullptr;
    const void* weights_device = nullptr;
    int ggml_type = 0;
    int token_id = 0;
    int hidden_dim = 0;
    StreamHandle stream = nullptr;
};

struct AttentionDecodeArgs {
    void* output = nullptr;
    const void* q = nullptr;
    const void* k_cache = nullptr;
    const void* v_cache = nullptr;
    int n_heads = 0;
    int n_kv_heads = 0;
    int head_dim = 0;
    int seq_len = 0;
    float scale = 1.0f;
    bool kv_int8 = false;
    const float* k_scales = nullptr;
    const float* v_scales = nullptr;
    StreamHandle stream = nullptr;
};

struct AttentionPrefillBatchedArgs {
    void* output = nullptr;
    const void* q = nullptr;
    const void* k_cache = nullptr;
    const void* v_cache = nullptr;
    int n_heads = 0;
    int n_kv_heads = 0;
    int head_dim = 0;
    int N = 0;
    int start_pos = 0;
    float scale = 1.0f;
    bool kv_int8 = false;
    const float* k_scales = nullptr;
    const float* v_scales = nullptr;
    StreamHandle stream = nullptr;
};

using GemvQuantFn = Status (*)(const GemvQuantArgs&);
using GemvQuantAddFn = Status (*)(const GemvQuantAddArgs&);
using GemvQuantPairFn = Status (*)(const GemvQuantPairArgs&);
using GemvQuantTripleFn = Status (*)(const GemvQuantTripleArgs&);
using GemmQuantBatchedFn = Status (*)(const GemmQuantBatchedArgs&);
using GemvQuantF32Fn = Status (*)(const GemvQuantF32Args&);
using DequantEmbeddingRowFn = Status (*)(const DequantEmbeddingRowArgs&);

struct Q4Kernels {
    GemvQuantFn gemv_quant = nullptr;
    GemvQuantAddFn gemv_quant_add = nullptr;
    GemvQuantPairFn gemv_quant_pair = nullptr;
    GemvQuantTripleFn gemv_quant_triple = nullptr;
    GemmQuantBatchedFn gemm_quant_batched = nullptr;
    GemvQuantF32Fn gemv_quant_f32 = nullptr;
    DequantEmbeddingRowFn dequant_embedding_row = nullptr;
};

constexpr const char* version_string() {
    return REFLEX_INFER_VERSION;
}

constexpr HardwareProfile orin_nano_8gb_sm87() {
    return {
        DeviceClass::OrinNano8GB,
        87,
        8,
        48u * 1024u,
        8ull * 1024ull * 1024ull * 1024ull,
        102.0f,
    };
}

constexpr KernelSupport query_support(
    const HardwareProfile&,
    const ModelShape&,
    const QuantDescriptor&) {
#if REFLEX_INFER_HAS_CUDA
    KernelSupport support{};
    support.q4_gemv = true;
    support.q4_mmq_prefill = true;
    support.decode_attention = true;
    return support;
#else
    return {};
#endif
}

void register_q4_backend(const Q4Kernels& kernels) noexcept;
Q4Kernels registered_q4_backend() noexcept;

Status gemv_quant(const GemvQuantArgs& args);
Status gemv_quant_add(const GemvQuantAddArgs& args);
Status gemv_quant_pair(const GemvQuantPairArgs& args);
Status gemv_quant_triple(const GemvQuantTripleArgs& args);
Status gemm_quant_batched(const GemmQuantBatchedArgs& args);
Status gemv_quant_f32(const GemvQuantF32Args& args);
Status dequant_embedding_row(const DequantEmbeddingRowArgs& args);

Status flash_attention_decode(const AttentionDecodeArgs& args);
Status flash_attention_prefill_batched(const AttentionPrefillBatchedArgs& args);

}  // namespace reflex::infer
