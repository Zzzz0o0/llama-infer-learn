#include "op/matmul.h"
#include "kernels/kernels_interface.h"
#include "base/base.h"
namespace op{
MatmulLayer::MatmulLayer(base::DeviceType device_type, int32_t dim0, int32_t dim1,
                        bool is_quant_layer, bool has_bias)
    : LayerParam(device_type, LayerType::kLayerMatmul, is_quant_layer, "Matmul"),
    dim0_(dim0),
    dim1_(dim1),
    has_bias_(has_bias){
        reset_input_size(1);
        reset_output_size(1);
        reset_weight_size(1);
        if(has_bias_){
            bias_.resize(1);
        }
}

base::Status MatmulLayer::check() const{
    return base::error::Success();
}

base::Status MatmulLayer::forward(){
    auto status = check();
    if(!status) return status;

    kernel::get_matmul_kernel(device_type_)(get_input(0), get_weight(0), get_output(0), 1.0);
    if(has_bias_){
        kernel::get_add_kernel(device_type_)(get_output(0), get_bias(0), get_output(0));
    }

    return base::error::Success();
}

base::Status MatmulLayer::set_bias(int32_t idx, int32_t& dim, const void* bias_ptr,
                                    base::DeviceType device_type){
    // size_t size = dim * sizeof(float);
    // std::shared_ptr<base::Buffer> buffer = std::make_shared<base::Buffer>(size, nullptr, const_cast<void*>(bias_ptr), true);
    // buffer->set_device_type(device_type);

    // tensor::Tensor bias(base::DataType::kDataTypeFp32, dim);
    // bias.set_device_type(device_type);
    // bias.assign(buffer);
    // bias_.at(idx) = bias;
    return base::error::Success();
}

base::Status MatmulLayer::set_bias(int32_t idx, const tensor::Tensor& bias){
    bias_.at(idx) = bias;
    return base::error::Success();
}

tensor::Tensor& MatmulLayer::get_bias(int32_t idx){
    return bias_.at(idx);
}

const tensor::Tensor& MatmulLayer::get_bias(int32_t idx) const{
    return bias_.at(idx);
}

} // namespace op