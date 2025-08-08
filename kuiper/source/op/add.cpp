#include "op/add.h"
#include "kernels/kernels_interface.h"
namespace op{
VecAddLayer::VecAddLayer(base::DeviceType device_type)
    :Layer(device_type, LayerType::kLayerAdd, "Add"){
        reset_input_size(2);
        reset_output_size(1);
}

base::Status VecAddLayer::check() const{
    return base::error::Success();
}

base::Status VecAddLayer::forward(){
    auto status = this->check();
    if(!status) return status;
    auto input1 = this->get_input(0);
    auto input2 = this->get_input(1);
    auto output = this->get_output(0);
    kernel::get_add_kernel(device_type_)(input1, input2, output);
    return base::error::Success();
}

} // namespace op