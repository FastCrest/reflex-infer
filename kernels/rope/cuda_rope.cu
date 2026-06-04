// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 FastCrest
//
// CUDA C++ reference for RoPE (rotary position embedding).
//
// One thread block per (batch, seq_pos, head_tile). Each thread handles a
// pair of dimensions (the rotation pair). Both interleaved and half-rotated
// layouts are supported via a template parameter.

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

template <typename T, bool INTERLEAVED>
__global__ void rope_kernel(
    const T* __restrict__ x,        // [B, S, H, D]
    T* __restrict__ y,              // [B, S, H, D]
    const float* __restrict__ cos,  // [S, D]
    const float* __restrict__ sin,  // [S, D]
    int B, int S, int H, int D
) {
    int b = blockIdx.x;
    int s = blockIdx.y;
    int h = blockIdx.z;
    int d = threadIdx.x;

    if (b >= B || s >= S || h >= H || d >= D) return;

    int64_t base = ((int64_t)b * S + s) * H * D + (int64_t)h * D;
    int64_t cs_base = (int64_t)s * D;

    float xv = to_f<T>(x[base + d]);
    float partner_v;
    float sign;

    if (INTERLEAVED) {
        int partner_d = (d % 2 == 0) ? (d + 1) : (d - 1);
        partner_v = to_f<T>(x[base + partner_d]);
        sign = (d % 2 == 0) ? -1.0f : 1.0f;
    } else {
        int half = D / 2;
        int partner_d = (d < half) ? (d + half) : (d - half);
        partner_v = to_f<T>(x[base + partner_d]);
        sign = (d < half) ? -1.0f : 1.0f;
    }

    float c = cos[cs_base + d];
    float si = sin[cs_base + d];
    float out = xv * c + sign * partner_v * si;
    y[base + d] = from_f<T>(out);
}

torch::Tensor apply_rope_cuda(
    torch::Tensor x,
    torch::Tensor cos,
    torch::Tensor sin,
    bool interleaved,
    bool inplace
) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(x.dim() == 4, "x must be [B, S, H, D]");
    int B = x.size(0);
    int S = x.size(1);
    int H = x.size(2);
    int D = x.size(3);
    TORCH_CHECK(D % 2 == 0, "head_dim must be even");
    TORCH_CHECK(D <= 1024, "head_dim > 1024 not supported by reference kernel");

    auto y = inplace ? x : torch::empty_like(x);
    auto cos_f = cos.to(torch::kFloat32);
    auto sin_f = sin.to(torch::kFloat32);

    dim3 grid(B, S, H);
    dim3 block(D);

    AT_DISPATCH_SWITCH(
        x.scalar_type(),
        "apply_rope_cuda",
        AT_DISPATCH_CASE(at::ScalarType::Half, [&] {
            if (interleaved) {
                rope_kernel<__half, true><<<grid, block>>>(
                    reinterpret_cast<const __half*>(x.data_ptr()),
                    reinterpret_cast<__half*>(y.data_ptr()),
                    cos_f.data_ptr<float>(),
                    sin_f.data_ptr<float>(),
                    B, S, H, D);
            } else {
                rope_kernel<__half, false><<<grid, block>>>(
                    reinterpret_cast<const __half*>(x.data_ptr()),
                    reinterpret_cast<__half*>(y.data_ptr()),
                    cos_f.data_ptr<float>(),
                    sin_f.data_ptr<float>(),
                    B, S, H, D);
            }
        })
        AT_DISPATCH_CASE(at::ScalarType::BFloat16, [&] {
            if (interleaved) {
                rope_kernel<__nv_bfloat16, true><<<grid, block>>>(
                    reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
                    reinterpret_cast<__nv_bfloat16*>(y.data_ptr()),
                    cos_f.data_ptr<float>(),
                    sin_f.data_ptr<float>(),
                    B, S, H, D);
            } else {
                rope_kernel<__nv_bfloat16, false><<<grid, block>>>(
                    reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
                    reinterpret_cast<__nv_bfloat16*>(y.data_ptr()),
                    cos_f.data_ptr<float>(),
                    sin_f.data_ptr<float>(),
                    B, S, H, D);
            }
        })
        AT_DISPATCH_CASE(at::ScalarType::Float, [&] {
            if (interleaved) {
                rope_kernel<float, true><<<grid, block>>>(
                    x.data_ptr<float>(),
                    y.data_ptr<float>(),
                    cos_f.data_ptr<float>(),
                    sin_f.data_ptr<float>(),
                    B, S, H, D);
            } else {
                rope_kernel<float, false><<<grid, block>>>(
                    x.data_ptr<float>(),
                    y.data_ptr<float>(),
                    cos_f.data_ptr<float>(),
                    sin_f.data_ptr<float>(),
                    B, S, H, D);
            }
        })
    );

    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("apply_rope_cuda", &apply_rope_cuda,
          "RoPE forward (CUDA)");
}
