#include "cpu/add_kernel.h"
#include "kernels_interface.h"
namespace kernel{
    AddKernel get_add_kernel(base::DeviceType device_type){
        return add_kernel_cpu;
    }
} // namespace kernel