#include "matmul_kernel.h"
#include <armadillo>
namespace kernel{
void matmul_kernel_cpu(const tensor::Tensor& input, const tensor::Tensor& weight,
                        const tensor::Tensor& output, float scale){

    const float* input_ptr = input.ptr<float>();
    const float* weight_ptr = weight.ptr<float>();
    const float* output_ptr = output.ptr<float>();

    int32_t in_dim1 = 1;
    int32_t in_dim0 = 1;
    if (input.dims().size() == 2) {
        in_dim0 = input.get_dim(0);
        in_dim1 = input.get_dim(1);
    } else if (input.dims().size() == 1) {
        in_dim0 = input.get_dim(0);
    }
    const int32_t wei_dim0 = weight.get_dim(0);
    const int32_t wei_dim1 = weight.get_dim(1);
    arma::fmat input_mat(const_cast<float*>(input_ptr), in_dim1, in_dim0, false, true);
    arma::fmat weight_mat(const_cast<float*>(weight_ptr), wei_dim1, wei_dim0, false, true);
    arma::fmat output_mat(const_cast<float*>(output_ptr), in_dim1, wei_dim0, false, true);
    output_mat = ((input_mat * weight_mat)) * scale;
}
} // namespace kernel