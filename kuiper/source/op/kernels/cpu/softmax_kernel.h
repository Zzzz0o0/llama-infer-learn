#ifndef LLAMA_INFER_SOFTMAX_KERNEL_H
#define LLAMA_INFER_SOFTMAX_KERNEL_H
#include "tensor/tensor.h"
namespace kernel {
void softmax_kernel_cpu(const tensor::Tensor& input);
}  // namespace kernel
#endif  // LLAMA_INFER_SOFTMAX_KERNEL_H
