// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 FastCrest
//
// CUDA C++ reference for paged KV cache operations.
//
// vLLM-style storage:
//   k_cache, v_cache: [num_blocks, num_heads, block_size, head_dim]
//   block_table:      [num_seqs, max_blocks_per_seq]
//   slot_mapping:     [N] flat slot index
//
// These are memory-movement kernels. They exist primarily as a fallback for
// devices where Triton isn't usable. The Triton path is the production
// implementation.

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

template <typename T>
__global__ void kv_paged_append_kernel(
    const T* __restrict__ k_in,
    const T* __restrict__ v_in,
    T* __restrict__ k_cache,
    T* __restrict__ v_cache,
    const int64_t* __restrict__ slot_mapping,
    int N,
    int num_heads,
    int head_dim,
    int block_size
) {
    int token_idx = blockIdx.x;
    int head_idx = blockIdx.y;
    int d = threadIdx.x;

    if (token_idx >= N || head_idx >= num_heads || d >= head_dim) return;

    int64_t slot = slot_mapping[token_idx];
    int64_t page_idx = slot / block_size;
    int64_t pos = slot % block_size;

    int64_t cache_offset = (
        page_idx * num_heads * block_size * head_dim
        + head_idx * block_size * head_dim
        + pos * head_dim
        + d
    );
    int64_t in_offset = (
        token_idx * num_heads * head_dim
        + head_idx * head_dim
        + d
    );

    k_cache[cache_offset] = k_in[in_offset];
    v_cache[cache_offset] = v_in[in_offset];
}

template <typename T>
__global__ void kv_paged_lookup_kernel(
    const T* __restrict__ k_cache,
    const T* __restrict__ v_cache,
    T* __restrict__ k_out,
    T* __restrict__ v_out,
    const int64_t* __restrict__ block_table,
    const int32_t* __restrict__ context_lens,
    int B,
    int num_heads,
    int max_ctx,
    int head_dim,
    int block_size,
    int max_blocks_per_seq
) {
    int seq_idx = blockIdx.x;
    int head_idx = blockIdx.y;
    int token_idx = blockIdx.z * blockDim.y + threadIdx.y;
    int d = threadIdx.x;

    if (seq_idx >= B || head_idx >= num_heads ||
        token_idx >= max_ctx || d >= head_dim) return;

    int ctx_len = context_lens[seq_idx];

    int64_t out_offset = (
        seq_idx * num_heads * max_ctx * head_dim
        + head_idx * max_ctx * head_dim
        + token_idx * head_dim
        + d
    );

    if (token_idx >= ctx_len) {
        k_out[out_offset] = T(0);
        v_out[out_offset] = T(0);
        return;
    }

    int blk_idx = token_idx / block_size;
    int pos = token_idx % block_size;
    int64_t page_idx = block_table[seq_idx * max_blocks_per_seq + blk_idx];

    int64_t cache_offset = (
        page_idx * num_heads * block_size * head_dim
        + head_idx * block_size * head_dim
        + pos * head_dim
        + d
    );

    k_out[out_offset] = k_cache[cache_offset];
    v_out[out_offset] = v_cache[cache_offset];
}

void kv_paged_append_cuda(
    torch::Tensor k_in,
    torch::Tensor v_in,
    torch::Tensor k_cache,
    torch::Tensor v_cache,
    torch::Tensor slot_mapping
) {
    TORCH_CHECK(k_in.is_cuda(), "k_in must be CUDA");
    int N = k_in.size(0);
    int num_heads = k_in.size(1);
    int head_dim = k_in.size(2);
    int block_size = k_cache.size(2);

    dim3 grid(N, num_heads);
    dim3 block(head_dim);

    AT_DISPATCH_SWITCH(
        k_in.scalar_type(),
        "kv_paged_append_cuda",
        AT_DISPATCH_CASE(at::ScalarType::Half, [&] {
            kv_paged_append_kernel<__half><<<grid, block>>>(
                reinterpret_cast<const __half*>(k_in.data_ptr()),
                reinterpret_cast<const __half*>(v_in.data_ptr()),
                reinterpret_cast<__half*>(k_cache.data_ptr()),
                reinterpret_cast<__half*>(v_cache.data_ptr()),
                slot_mapping.data_ptr<int64_t>(),
                N, num_heads, head_dim, block_size);
        })
        AT_DISPATCH_CASE(at::ScalarType::BFloat16, [&] {
            kv_paged_append_kernel<__nv_bfloat16><<<grid, block>>>(
                reinterpret_cast<const __nv_bfloat16*>(k_in.data_ptr()),
                reinterpret_cast<const __nv_bfloat16*>(v_in.data_ptr()),
                reinterpret_cast<__nv_bfloat16*>(k_cache.data_ptr()),
                reinterpret_cast<__nv_bfloat16*>(v_cache.data_ptr()),
                slot_mapping.data_ptr<int64_t>(),
                N, num_heads, head_dim, block_size);
        })
    );
}

std::vector<torch::Tensor> kv_paged_lookup_cuda(
    torch::Tensor k_cache,
    torch::Tensor v_cache,
    torch::Tensor block_table,
    torch::Tensor context_lens,
    int max_ctx
) {
    TORCH_CHECK(k_cache.is_cuda(), "k_cache must be CUDA");
    int B = context_lens.size(0);
    int num_heads = k_cache.size(1);
    int block_size = k_cache.size(2);
    int head_dim = k_cache.size(3);
    int max_blocks_per_seq = block_table.size(1);

    auto opts = k_cache.options();
    auto k_out = torch::zeros({B, num_heads, max_ctx, head_dim}, opts);
    auto v_out = torch::zeros({B, num_heads, max_ctx, head_dim}, opts);

    dim3 block(head_dim, 4);
    dim3 grid(B, num_heads, (max_ctx + 3) / 4);

    AT_DISPATCH_SWITCH(
        k_cache.scalar_type(),
        "kv_paged_lookup_cuda",
        AT_DISPATCH_CASE(at::ScalarType::Half, [&] {
            kv_paged_lookup_kernel<__half><<<grid, block>>>(
                reinterpret_cast<const __half*>(k_cache.data_ptr()),
                reinterpret_cast<const __half*>(v_cache.data_ptr()),
                reinterpret_cast<__half*>(k_out.data_ptr()),
                reinterpret_cast<__half*>(v_out.data_ptr()),
                block_table.data_ptr<int64_t>(),
                context_lens.data_ptr<int32_t>(),
                B, num_heads, max_ctx, head_dim, block_size,
                max_blocks_per_seq);
        })
        AT_DISPATCH_CASE(at::ScalarType::BFloat16, [&] {
            kv_paged_lookup_kernel<__nv_bfloat16><<<grid, block>>>(
                reinterpret_cast<const __nv_bfloat16*>(k_cache.data_ptr()),
                reinterpret_cast<const __nv_bfloat16*>(v_cache.data_ptr()),
                reinterpret_cast<__nv_bfloat16*>(k_out.data_ptr()),
                reinterpret_cast<__nv_bfloat16*>(v_out.data_ptr()),
                block_table.data_ptr<int64_t>(),
                context_lens.data_ptr<int32_t>(),
                B, num_heads, max_ctx, head_dim, block_size,
                max_blocks_per_seq);
        })
    );

    return {k_out, v_out};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("kv_paged_append_cuda",
          &kv_paged_append_cuda,
          "Append into paged KV cache (CUDA)");
    m.def("kv_paged_lookup_cuda",
          &kv_paged_lookup_cuda,
          "Gather from paged KV cache (CUDA)");
}
