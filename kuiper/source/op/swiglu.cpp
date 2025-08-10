#include "op/swiglu.h"
#include "kernels/kernels_interface.h"

namespace op{
SwiGLULayer::SwiGLULayer(base::DeviceType device_type, int32_t hidden_dim)
    :Layer(device_type, LayerType::kLayerSwiGLU, "SwiGLU"), hidden_dim_(hidden_dim){
        reset_input_size(2);
        reset_output_size(1);
}

base::Status SwiGLULayer::check() const {
    return base::error::Success();
}

base::Status SwiGLULayer::forward(){
    auto status = check();
    if(!status) return status;

    auto input1 = this->get_input(0);
    auto input2 = this->get_input(1);
    auto output = this->get_output(0);

    kernel::get_swiglu_kernel(device_type_)(input1, input2, output);
    return base::error::Success();
}
} // namespace op