#ifndef KUIPER_OP_KERNELS_CPU_EMB_KERNEL_H_
#define KUIPER_OP_KERNELS_CPU_EMB_KERNEL_H_
#include "tensor/tensor.h"
namespace kernel{
void emb_kernel_cpu(const tensor::Tensor& input, const tensor::Tensor& weight,
                       const tensor::Tensor& output, int32_t vocab_size);
} // namespace kernel
#endif // KUIPER_OP_KERNELS_CPU_EMB_KERNEL_H_