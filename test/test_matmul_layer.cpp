#include <gtest/gtest.h>
#include "base/base.h"
#include "tensor/tensor.h"
#include "op/matmul.h"

TEST(test_matmul_cpu, matmul1){
    auto alloc = base::CPUDeviceAllocatorFactory::get_instance();
    int32_t m,n,k;
    m = 2, k = 3, n = 2;
    tensor::Tensor input(base::DataType::kDataTypeFp32, k, n, true, alloc);
    tensor::Tensor weight(base::DataType::kDataTypeFp32, m, k, true, alloc);
    tensor::Tensor out(base::DataType::kDataTypeFp32, m, n, true, alloc);

    for(int i=1;i<=6;++i){
        input.index<float>(i-1) = i * 1.0;
        weight.index<float>(i-1) = i * 1.0;
    }


    op::MatmulLayer matmul_layer(base::DeviceType::kDeviceCPU, 0, 0, false, false);
    matmul_layer.set_input(0, input);
    matmul_layer.set_weight(0, weight);
    matmul_layer.set_output(0, out);
    matmul_layer.forward();
    for(int i=0;i<4;++i){
        std::cout<<out.index<float>(i)<<std::endl;
    }
    float t = out.index<float>(0);
    ASSERT_EQ(t, 22);
    
}