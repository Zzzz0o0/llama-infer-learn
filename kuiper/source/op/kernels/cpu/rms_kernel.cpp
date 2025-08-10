#include "rms_kernel.h"
#include <armadillo>
namespace kernel{
void rms_kernel_cpu(const tensor::Tensor& input, const tensor::Tensor& weight,
                        const tensor::Tensor& output){

    const float* input_ptr = input.ptr<float>();
    const float* weight_ptr = weight.ptr<float>();
    const float* output_ptr = output.ptr<float>();

    const int32_t dim = static_cast<int32_t>(input.size());

    arma::fvec in_tensor(const_cast<float*>(input_ptr), dim, false, true);
    arma::fvec out_tensor(const_cast<float*>(output_ptr), dim, false, true);
    arma::fvec wei_tensor(const_cast<float*>(weight_ptr), dim, false, true);

    const float eps = 1e-6f;

    const float mean = arma::as_scalar(arma::mean(arma::pow(in_tensor, 2))) + eps;
    const float rsqrt = 1.f / std::sqrt(mean);
    out_tensor = wei_tensor % (rsqrt * in_tensor);
}
} // namespace kernel