#include <gtest/gtest.h>
#include "base/base.h"
#include "tensor/tensor.h"
#include "op/rmsnorm.h"

TEST(test_rmsnorm_cpu, rmsnorm1){
    auto alloc = base::CPUDeviceAllocatorFactory::get_instance();
    int32_t size = 5;
    tensor::Tensor input(base::DataType::kDataTypeFp32, size, true, alloc);
    tensor::Tensor weight(base::DataType::kDataTypeFp32, size, true, alloc);
    tensor::Tensor out(base::DataType::kDataTypeFp32, size, true, alloc);

    for(int i=0;i<size;++i){
        input.index<float>(i) = i * 2.0;
        weight.index<float>(i) = i * 1.0;
    }


    op::RmsNormLayer rmsnorm_layer(base::DeviceType::kDeviceCPU, size);
    rmsnorm_layer.set_input(0, input);
    rmsnorm_layer.set_weight(0, weight);
    rmsnorm_layer.set_output(0, out);
    rmsnorm_layer.forward();
    for(int i=0;i<size;++i){
        std::cout<<out.index<float>(i)<<std::endl;
    }

    ASSERT_EQ(out.index<float>(0), 0);
    
}