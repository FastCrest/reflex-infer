#pragma once

#include <cstddef>
#include <cstdint>

#ifndef REFLEX_INFER_VERSION
#define REFLEX_INFER_VERSION "0.1.0"
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
    // Capability discovery is wired before kernels move out of reflex-llm.
    // Return no fast paths until each external kernel is implemented and
    // validated against the runtime fallback.
    return {};
}

}  // namespace reflex::infer

