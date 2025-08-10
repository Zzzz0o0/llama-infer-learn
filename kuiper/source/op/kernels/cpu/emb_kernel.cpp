#include "emb_kernel.h"
#include <cstring>
namespace kernel{

void emb_kernel_cpu(const tensor::Tensor& input, const tensor::Tensor& weight,
                       const tensor::Tensor& output, int32_t vocab_size){
    const int32_t input_num = static_cast<int32_t>(input.size());
    const int32_t weight_dim = weight.get_dim(1);

    for(int32_t i=0;i<input_num; ++i){
        int32_t token = static_cast<int32_t>(input.index<float>(i));
        float* dest_ptr = const_cast<float*>(output.ptr<float>(i*weight_dim));
        float* src_ptr = const_cast<float*>(weight.ptr<float>(token * weight_dim));
        std::memcpy(dest_ptr, src_ptr,  weight_dim * sizeof(float));
    }
}

} // namespace kernel