#ifndef KUIPER_KERNELS_CPU_RMS_KERNEL_H_
#define KUIPER_KERNELS_CPU_RMS_KERNEL_H_
#include "tensor/tensor.h"
namespace kernel{
void rms_kernel_cpu(const tensor::Tensor& input, const tensor::Tensor& weight,
                    const tensor::Tensor& output);
}

#endif // KUIPER_KERNELS_CPU_RMS_KERNEL_H_