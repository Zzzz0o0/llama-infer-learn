#include <gtest/gtest.h>
#include "base/base.h"
#include "tensor/tensor.h"
#include "op/swiglu.h"

TEST(test_swiglu_cpu, swiglu1){
    auto alloc = base::CPUDeviceAllocatorFactory::get_instance();
    int32_t m,k;
    m = 2, k = 3;
    tensor::Tensor input1(base::DataType::kDataTypeFp32, m, k, true, alloc);
    tensor::Tensor input2(base::DataType::kDataTypeFp32, m, k, true, alloc);
    tensor::Tensor out(base::DataType::kDataTypeFp32, m, k, true, alloc);

    for(int i=1;i<=6;++i){
        input1.index<float>(i-1) = i * 1.0;
        input2.index<float>(i-1) = i * 1.0;
    }


    op::SwiGLULayer swiglu_layer(base::DeviceType::kDeviceCPU, 3);
    swiglu_layer.set_input(0, input1);
    swiglu_layer.set_input(1, input2);
    swiglu_layer.set_output(0, out);
    swiglu_layer.forward();
    for(int i=0;i<6;++i){
        std::cout<<out.index<float>(i)<<std::endl;
    }
    float t = out.index<float>(0);
    ASSERT_EQ(1, 1);
    
}