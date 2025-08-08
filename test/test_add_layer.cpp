#include <gtest/gtest.h>
#include "base/base.h"
#include "tensor/tensor.h"
#include "op/add.h"
TEST(test_add_cpu, add1){
    auto alloc = base::CPUDeviceAllocatorFactory::get_instance();
    int32_t size = 3;
    tensor::Tensor t1(base::DataType::kDataTypeFp32, size, true, alloc);
    tensor::Tensor t2(base::DataType::kDataTypeFp32, size, true, alloc);
    tensor::Tensor out(base::DataType::kDataTypeFp32, size, true, alloc);

    for(int i=0;i<size;++i){
        t1.index<float>(i) = i * 1.0;
        t2.index<float>(i) = i * 2.0;
    }

    op::VecAddLayer add_layer(base::DeviceType::kDeviceCPU);
    add_layer.set_input(0, t1);
    add_layer.set_input(1, t2);
    add_layer.set_output(0, out);
    add_layer.forward();

    for(int i=0;i<size;++i){
        float t = out.index<float>(i);
        ASSERT_EQ(t, i+i*2.0);
    }
}