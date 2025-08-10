#ifndef KUIPER_KERNELS_CPU_MATMUL_KERNEL_H_
#define KUIPER_KERNELS_CPU_MATMUL_KERNEL_H_
#include "tensor/tensor.h"
namespace kernel{
void matmul_kernel_cpu(const tensor::Tensor& input, const tensor::Tensor& weight,
                    const tensor::Tensor& output, float scale);
}

#endif // KUIPER_KERNELS_CPU_MATMUL_KERNEL_H_