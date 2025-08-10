#include "op/rmsnorm.h"
#include "kernels/kernels_interface.h"
namespace op{
RmsNormLayer::RmsNormLayer(base::DeviceType device_type, int32_t dim)
    :LayerParam(device_type, LayerType::kLayerRMSNorm, false, "RMSNorm"), dim_(dim){
    reset_input_size(1);
    reset_output_size(1);
    reset_weight_size(1);
}

base::Status RmsNormLayer::check() const {
    return base::error::Success();
}

base::Status RmsNormLayer::forward() {
    auto status = check();
    if(!status) return status;
    auto input = this->get_input(0);
    auto weight = this->get_weight(0);
    auto output = this->get_output(0);

    kernel::get_rmsnorm_kernel(device_type_)(input, weight, output);
    return base::error::Success();
}
}