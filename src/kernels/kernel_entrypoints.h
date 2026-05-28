#pragma once

#include "reflex/infer.h"

namespace reflex::infer::kernels {

Status gemv_quant(const GemvQuantArgs& args);
Status gemv_quant_add(const GemvQuantAddArgs& args);
Status gemv_quant_pair(const GemvQuantPairArgs& args);
Status gemv_quant_triple(const GemvQuantTripleArgs& args);
Status gemm_quant_batched(const GemmQuantBatchedArgs& args);
Status gemv_quant_f32(const GemvQuantF32Args& args);
Status dequant_embedding_row(const DequantEmbeddingRowArgs& args);

Status flash_attention_decode(const AttentionDecodeArgs& args);
Status flash_attention_prefill_batched(const AttentionPrefillBatchedArgs& args);

}  // namespace reflex::infer::kernels
