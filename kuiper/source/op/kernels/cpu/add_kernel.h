#ifndef KUIPER_KERNELS_CPU_ADD_KERNEL_H_
#define KUIPER_KERNELS_CPU_ADD_KERNEL_H_
#include "tensor/tensor.h"
namespace kernel{
void add_kernel_cpu(const tensor::Tensor& input1, const tensor::Tensor& input2,
                    const tensor::Tensor& output);
}

#endif // KUIPER_KERNELS_CPU_ADD_KERNEL_H_