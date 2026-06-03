// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 FastCrest
//
// CUDA C++ reference for fused Linear + (Layer | RMS) Norm.
//
// Strategy:
//   * One CUDA block per row.
//   * Block-strided GEMV into a per-row buffer in shared memory.
//   * Block-wide reduction to compute mean / variance / RMS.
//   * Write normalized output.
//
// This is the reference path; the Triton kernel is the production one.

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
template <> __device__ __forceinline__ float to_f<float>(float x) {
    return x;
}

template <typename T>
__device__ __forceinline__ T from_f(float x);
template <> __device__ __forceinline__ __half from_f<__half>(float x) {
    return __float2half(x);
}
template <> __device__ __forceinline__ __nv_bfloat16
from_f<__nv_bfloat16>(float x) {
    return __float2bfloat16(x);
}
template <> __device__ __forceinline__ float from_f<float>(float x) {
    return x;
}

// Block reduce to a single value across threadIdx.x. Standard tree reduction
// over shared memory; faster paths (warp shfl) are left to the Triton kernel.
__device__ __forceinline__ float block_reduce_sum(float v, float* smem) {
    int tid = threadIdx.x;
    smem[tid] = v;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    return smem[0];
}

template <typename T, int NORM_TYPE, bool HAS_BIAS, bool HAS_AFFINE>
__global__ void fused_linear_norm_kernel(
    const T* __restrict__ x,         // [N, D_in]
    const T* __restrict__ w,         // [D_out, D_in]
    const T* __restrict__ bias,
    const T* __restrict__ gamma,
    const T* __restrict__ beta,
    T* __restrict__ y,
    int N, int D_in, int D_out, float eps
) {
    int row = blockIdx.x;
    if (row >= N) return;

    extern __shared__ float smem[];
    float* row_out = smem;                  // [D_out]
    float* reduce_buf = smem + D_out;       // [blockDim.x]

    int tid = threadIdx.x;

    // ---- GEMV: row_out[o] = sum_k x[row, k] * w[o, k] (+ bias[o]) ----
    for (int o = tid; o < D_out; o += blockDim.x) {
        float acc = 0.0f;
        for (int k = 0; k < D_in; ++k) {
            acc += to_f<T>(x[row * D_in + k]) * to_f<T>(w[o * D_in + k]);
        }
        if (HAS_BIAS) acc += to_f<T>(bias[o]);
        row_out[o] = acc;
    }
    __syncthreads();

    // ---- Normalization ----
    if (NORM_TYPE == 0) {
        // LayerNorm: mean, variance.
        float local_sum = 0.0f;
        for (int o = tid; o < D_out; o += blockDim.x) local_sum += row_out[o];
        float total = block_reduce_sum(local_sum, reduce_buf);
        float mean = total / D_out;

        float local_sq = 0.0f;
        for (int o = tid; o < D_out; o += blockDim.x) {
            float c = row_out[o] - mean;
            local_sq += c * c;
        }
        float total_sq = block_reduce_sum(local_sq, reduce_buf);
        float var = total_sq / D_out;
        float rstd = rsqrtf(var + eps);

        for (int o = tid; o < D_out; o += blockDim.x) {
            float v = (row_out[o] - mean) * rstd;
            if (HAS_AFFINE) {
                v = v * to_f<T>(gamma[o]) + to_f<T>(beta[o]);
            }
            y[row * D_out + o] = from_f<T>(v);
        }
    } else {
        // RMSNorm.
        float local_sq = 0.0f;
        for (int o = tid; o < D_out; o += blockDim.x) {
            local_sq += row_out[o] * row_out[o];
        }
        float total_sq = block_reduce_sum(local_sq, reduce_buf);
        float mean_sq = total_sq / D_out;
        float rstd = rsqrtf(mean_sq + eps);

        for (int o = tid; o < D_out; o += blockDim.x) {
            float v = row_out[o] * rstd;
            if (HAS_AFFINE) v = v * to_f<T>(gamma[o]);
            y[row * D_out + o] = from_f<T>(v);
        }
    }
}

torch::Tensor fused_linear_norm_cuda(
    torch::Tensor x,
    torch::Tensor w,
    torch::optional<torch::Tensor> bias_opt,
    torch::optional<torch::Tensor> gamma_opt,
    torch::optional<torch::Tensor> beta_opt,
    double eps,
    int64_t norm_type
) {
    auto x_2d = x.contiguous().view({-1, x.size(-1)});
    int N = x_2d.size(0);
    int D_in = x_2d.size(1);
    int D_out = w.size(0);
    auto y = torch::empty({N, D_out}, x.options());

    bool has_bias = bias_opt.has_value();
    bool has_affine = gamma_opt.has_value();
    auto bias = has_bias ? bias_opt.value() : x_2d;
    auto gamma = has_affine ? gamma_opt.value() : x_2d;
    auto beta = (has_affine && beta_opt.has_value()) ? beta_opt.value() : gamma;

    int block_size = 128;
    dim3 grid(N);
    dim3 block(block_size);
    size_t smem_bytes = (D_out + block_size) * sizeof(float);

    #define DISPATCH(SCALAR_TYPE, T)                                          \
        AT_DISPATCH_CASE(SCALAR_TYPE, [&] {                                   \
            if (norm_type == 0) {                                             \
                if (has_bias && has_affine) {                                 \
                    fused_linear_norm_kernel<T, 0, true, true>                \
                        <<<grid, block, smem_bytes>>>(                        \
                            reinterpret_cast<const T*>(x_2d.data_ptr()),      \
                            reinterpret_cast<const T*>(w.data_ptr()),         \
                            reinterpret_cast<const T*>(bias.data_ptr()),      \
                            reinterpret_cast<const T*>(gamma.data_ptr()),     \
                            reinterpret_cast<const T*>(beta.data_ptr()),      \
                            reinterpret_cast<T*>(y.data_ptr()),               \
                            N, D_in, D_out, static_cast<float>(eps));         \
                } else if (has_bias) {                                        \
                    fused_linear_norm_kernel<T, 0, true, false>               \
                        <<<grid, block, smem_bytes>>>(                        \
                            reinterpret_cast<const T*>(x_2d.data_ptr()),      \
                            reinterpret_cast<const T*>(w.data_ptr()),         \
                            reinterpret_cast<const T*>(bias.data_ptr()),      \
                            nullptr, nullptr,                                  \
                            reinterpret_cast<T*>(y.data_ptr()),               \
                            N, D_in, D_out, static_cast<float>(eps));         \
                } else {                                                      \
                    fused_linear_norm_kernel<T, 0, false, false>              \
                        <<<grid, block, smem_bytes>>>(                        \
                            reinterpret_cast<const T*>(x_2d.data_ptr()),      \
                            reinterpret_cast<const T*>(w.data_ptr()),         \
                            nullptr, nullptr, nullptr,                        \
                            reinterpret_cast<T*>(y.data_ptr()),               \
                            N, D_in, D_out, static_cast<float>(eps));         \
                }                                                             \
            } else {                                                          \
                if (has_affine) {                                             \
                    fused_linear_norm_kernel<T, 1, false, true>               \
                        <<<grid, block, smem_bytes>>>(                        \
                            reinterpret_cast<const T*>(x_2d.data_ptr()),      \
                            reinterpret_cast<const T*>(w.data_ptr()),         \
                            nullptr,                                          \
                            reinterpret_cast<const T*>(gamma.data_ptr()),     \
                            nullptr,                                          \
                            reinterpret_cast<T*>(y.data_ptr()),               \
                            N, D_in, D_out, static_cast<float>(eps));         \
                } else {                                                      \
                    fused_linear_norm_kernel<T, 1, false, false>              \
                        <<<grid, block, smem_bytes>>>(                        \
                            reinterpret_cast<const T*>(x_2d.data_ptr()),      \
                            reinterpret_cast<const T*>(w.data_ptr()),         \
                            nullptr, nullptr, nullptr,                        \
                            reinterpret_cast<T*>(y.data_ptr()),               \
                            N, D_in, D_out, static_cast<float>(eps));         \
                }                                                             \
            }                                                                 \
        })

    AT_DISPATCH_SWITCH(
        x.scalar_type(),
        "fused_linear_norm_cuda",
        DISPATCH(at::ScalarType::Half, __half)
        DISPATCH(at::ScalarType::BFloat16, __nv_bfloat16)
        DISPATCH(at::ScalarType::Float, float)
    );

    auto out_shape = x.sizes().vec();
    out_shape.back() = D_out;
    return y.view(out_shape);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_linear_norm_cuda",
          &fused_linear_norm_cuda,
          "Fused Linear + LayerNorm/RMSNorm (CUDA)");
}
