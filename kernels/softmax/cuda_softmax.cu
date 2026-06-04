// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 FastCrest
//
// CUDA C++ reference for online softmax.
//
// Single block per row, two-pass online softmax. The Triton kernel is the
// production path; this lives here for the Triton-less fallback and for
// numerical cross-checking.

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

template <typename T>
__device__ __forceinline__ float to_f(T x);
template <> __device__ __forceinline__ float to_f<__half>(__half x) {
    return __half2float(x);
}
template <> __device__ __forceinline__ float to_f<__nv_bfloat16>(__nv_bfloat16 x) {
    return __bfloat162float(x);
}
template <> __device__ __forceinline__ float to_f<float>(float x) { return x; }

template <typename T>
__device__ __forceinline__ T from_f(float x);
template <> __device__ __forceinline__ __half from_f<__half>(float x) {
    return __float2half(x);
}
template <> __device__ __forceinline__ __nv_bfloat16
from_f<__nv_bfloat16>(float x) {
    return __float2bfloat16(x);
}
template <> __device__ __forceinline__ float from_f<float>(float x) { return x; }

template <typename T, bool IS_CAUSAL, bool IS_LOG>
__global__ void online_softmax_kernel(
    const T* __restrict__ x, T* __restrict__ y,
    int M, int N
) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int block_size = blockDim.x;

    extern __shared__ float smem[];

    // Pass 1: find row max (warp-level reduction with shared mem).
    float local_max = -INFINITY;
    for (int j = tid; j < N; j += block_size) {
        float v = to_f<T>(x[row * N + j]);
        if (IS_CAUSAL && j > row) v = -INFINITY;
        if (v > local_max) local_max = v;
    }
    smem[tid] = local_max;
    __syncthreads();
    for (int s = block_size / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] = fmaxf(smem[tid], smem[tid + s]);
        __syncthreads();
    }
    float row_max = smem[0];

    // Pass 2: compute denominator.
    float local_sum = 0.0f;
    for (int j = tid; j < N; j += block_size) {
        float v = to_f<T>(x[row * N + j]);
        if (IS_CAUSAL && j > row) v = -INFINITY;
        local_sum += expf(v - row_max);
    }
    smem[tid] = local_sum;
    __syncthreads();
    for (int s = block_size / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    float row_sum = smem[0];
    float log_sum = logf(row_sum);

    // Pass 3: write softmax / log_softmax.
    for (int j = tid; j < N; j += block_size) {
        float v = to_f<T>(x[row * N + j]);
        if (IS_CAUSAL && j > row) v = -INFINITY;
        float out;
        if (IS_LOG) {
            out = v - row_max - log_sum;
        } else {
            out = expf(v - row_max) / row_sum;
        }
        y[row * N + j] = from_f<T>(out);
    }
}

torch::Tensor online_softmax_cuda(
    torch::Tensor x,
    bool is_causal,
    bool log
) {
    auto x_2d = x.contiguous().view({-1, x.size(-1)});
    int M = x_2d.size(0);
    int N = x_2d.size(1);
    auto y = torch::empty_like(x_2d);

    int block_size = 256;
    if (N < 256) block_size = 64;
    dim3 grid(M);
    dim3 block(block_size);
    size_t smem_bytes = block_size * sizeof(float);

    #define DISPATCH(SCALAR_TYPE, T)                                          \
        AT_DISPATCH_CASE(SCALAR_TYPE, [&] {                                   \
            if (is_causal && log) {                                           \
                online_softmax_kernel<T, true, true>                          \
                    <<<grid, block, smem_bytes>>>(                            \
                        reinterpret_cast<const T*>(x_2d.data_ptr()),          \
                        reinterpret_cast<T*>(y.data_ptr()), M, N);            \
            } else if (is_causal) {                                           \
                online_softmax_kernel<T, true, false>                         \
                    <<<grid, block, smem_bytes>>>(                            \
                        reinterpret_cast<const T*>(x_2d.data_ptr()),          \
                        reinterpret_cast<T*>(y.data_ptr()), M, N);            \
            } else if (log) {                                                 \
                online_softmax_kernel<T, false, true>                         \
                    <<<grid, block, smem_bytes>>>(                            \
                        reinterpret_cast<const T*>(x_2d.data_ptr()),          \
                        reinterpret_cast<T*>(y.data_ptr()), M, N);            \
            } else {                                                          \
                online_softmax_kernel<T, false, false>                        \
                    <<<grid, block, smem_bytes>>>(                            \
                        reinterpret_cast<const T*>(x_2d.data_ptr()),          \
                        reinterpret_cast<T*>(y.data_ptr()), M, N);            \
            }                                                                 \
        })

    AT_DISPATCH_SWITCH(
        x.scalar_type(),
        "online_softmax_cuda",
        DISPATCH(at::ScalarType::Half, __half)
        DISPATCH(at::ScalarType::BFloat16, __nv_bfloat16)
        DISPATCH(at::ScalarType::Float, float)
    );

    return y.view(x.sizes());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("online_softmax_cuda", &online_softmax_cuda,
          "Online softmax (CUDA)");
}
