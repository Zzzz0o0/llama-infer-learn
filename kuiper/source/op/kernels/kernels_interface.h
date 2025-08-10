#ifndef KERNELS_INTERFACE_H_
#define KERNELS_INTERFACE_H_
#include "tensor/tensor.h"
namespace kernel{
typedef void(*AddKernel)(const tensor::Tensor& input1, const tensor::Tensor& input2,
                        const tensor::Tensor& output);

using MatmulKernel = void(*)(const tensor::Tensor& input, const tensor::Tensor& weight,
                            const tensor::Tensor& output, float scale);

using EmbeddingKernel = void(*)(const tensor::Tensor& input, const tensor::Tensor& weight,
                                const tensor::Tensor& output, int32_t vocab_size);

using RMSNormKernel = void(*)(const tensor::Tensor& input, const tensor::Tensor& weight,
                                const tensor::Tensor& output);
                                
using RoPEKernel = void(*)(int32_t dim, int32_t kv_dim, int32_t head_size, const tensor::Tensor& input_q,
                            const tensor::Tensor& input_k, const tensor::Tensor& input_pos, const tensor::Tensor& sin_cache,
                            const tensor::Tensor& cos_cache);

using SwiGLUKernel = void(*)(const tensor::Tensor& input1, const tensor::Tensor& input2, const tensor::Tensor& output);

using MHAKernel = void(*)(int32_t pos, int32_t head_num, int32_t layer_index, int32_t seq_len,
                          int32_t kv_dim, int32_t kv_mul, int32_t head_size,
                          const tensor::Tensor& mha_out, const tensor::Tensor& query_tensor,
                          const tensor::Tensor& score_tensor,
                          const tensor::Tensor& key_cache_tensor,
                          const tensor::Tensor& value_cache_tensor, base::DeviceType device_type);

using SoftmaxKernel = void(*)(const tensor::Tensor& input);

using ScaleSumKernel = void(*)(const tensor::Tensor& value, const tensor::Tensor& scale, 
                                const tensor::Tensor& output, int t, int size, int stride);


AddKernel get_add_kernel(base::DeviceType device_type);

MatmulKernel get_matmul_kernel(base::DeviceType device_type);

EmbeddingKernel get_emb_kernel(base::DeviceType device_type);

RMSNormKernel get_rmsnorm_kernel(base::DeviceType device_type);

RoPEKernel get_rope_kernel(base::DeviceType device_type);

SwiGLUKernel get_swiglu_kernel(base::DeviceType device_type);

MHAKernel get_mha_kernel(base::DeviceType device_type);

SoftmaxKernel get_softmax_kernel(base::DeviceType device_type);

ScaleSumKernel get_scale_sum_kernel(base::DeviceType device_type);

} // namespace kernels

#endif // KERNELS_INTERFACE_H_