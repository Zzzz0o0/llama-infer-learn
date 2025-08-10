#include "op/rope.h"
#include "kernels/kernels_interface.h"
namespace op{
RoPELayer::RoPELayer(base::DeviceType device_type, int32_t dim, int32_t kv_dim, int32_t head_size)
    :Layer(device_type, LayerType::kLayerRoPE, "RoPe"),
    dim_(dim),
    kv_dim_(kv_dim),
    head_size_(head_size){
        reset_input_size(5);
        reset_output_size(1);
}

base::Status RoPELayer::check() const {
    return base::error::Success();
}

base::Status RoPELayer::forward(){
    auto status = check();
    if(!status) return status;

    tensor::Tensor input_q = this->get_input(0);
    tensor::Tensor input_k = this->get_input(1);
    tensor::Tensor input_pos = this->get_input(2);

    tensor::Tensor sin_cache = this->get_input(3);
    tensor::Tensor cos_cache = this->get_input(4);

    kernel::get_rope_kernel(device_type_)(dim_, kv_dim_, head_size_, input_q, input_k, 
                            input_pos, sin_cache, cos_cache);

    return base::error::Success();
}


} // namespace op