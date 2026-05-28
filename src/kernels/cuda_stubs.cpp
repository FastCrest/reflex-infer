#include "kernels/kernel_entrypoints.h"

namespace reflex::infer::kernels {

Status gemv_quant(const GemvQuantArgs&) {
    return Status::NotImplemented;
}

Status gemv_quant_add(const GemvQuantAddArgs&) {
    return Status::NotImplemented;
}

Status gemv_quant_pair(const GemvQuantPairArgs&) {
    return Status::NotImplemented;
}

Status gemv_quant_triple(const GemvQuantTripleArgs&) {
    return Status::NotImplemented;
}

Status gemm_quant_batched(const GemmQuantBatchedArgs&) {
    return Status::NotImplemented;
}

Status gemv_quant_f32(const GemvQuantF32Args&) {
    return Status::NotImplemented;
}

Status dequant_embedding_row(const DequantEmbeddingRowArgs&) {
    return Status::NotImplemented;
}

Status flash_attention_decode(const AttentionDecodeArgs&) {
    return Status::NotImplemented;
}

Status flash_attention_prefill_batched(const AttentionPrefillBatchedArgs&) {
    return Status::NotImplemented;
}

}  // namespace reflex::infer::kernels
