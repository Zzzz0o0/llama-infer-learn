#include "cpu/add_kernel.h"
#include "cpu/matmul_kernel.h"
#include "cpu/emb_kernel.h"
#include "cpu/rms_kernel.h"
#include "cpu/rope_kernel.h"
#include "cpu/swiglu_kernel.h"
#include "cpu/mha_kernel.h"
#include "cpu/softmax_kernel.h"
#include "cpu/scale_sum_kernel.h"
#include "kernels_interface.h"

namespace kernel{
    AddKernel get_add_kernel(base::DeviceType device_type){
        return add_kernel_cpu;
    }
    MatmulKernel get_matmul_kernel(base::DeviceType device_type){
        return matmul_kernel_cpu;
    }
    EmbeddingKernel get_emb_kernel(base::DeviceType device_type){
        return emb_kernel_cpu;
    }
    RMSNormKernel get_rmsnorm_kernel(base::DeviceType device_type){
        return rms_kernel_cpu;
    }
    RoPEKernel get_rope_kernel(base::DeviceType device_type){
        return rope_kernel_cpu;
    }
    SwiGLUKernel get_swiglu_kernel(base::DeviceType device_type){
        return swiglu_kernel_cpu;
    }
    MHAKernel get_mha_kernel(base::DeviceType device_type){
        return mha_kernel_cpu;
    }
    SoftmaxKernel get_softmax_kernel(base::DeviceType device_type){
        return softmax_kernel_cpu;
    }
    ScaleSumKernel get_scale_sum_kernel(base::DeviceType device_type){
        return scale_sum_kernel_cpu;
    }

} // namespace kernel