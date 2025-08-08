#ifndef KERNELS_INTERFACE_H_
#define KERNELS_INTERFACE_H_
#include "tensor/tensor.h"
namespace kernel{
typedef void(*AddKernel)(const tensor::Tensor& input1, const tensor::Tensor& input2,
                        const tensor::Tensor& output);

AddKernel get_add_kernel(base::DeviceType device_type);

} // namespace kernels

#endif // KERNELS_INTERFACE_H_