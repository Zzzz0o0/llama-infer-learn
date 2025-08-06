#include "tensor/tensor.h"
#include <gtest/gtest.h>
TEST(test_tensor, create){
    using namespace base;
    auto alloc = CPUDeviceAllocatorFactory::get_instance();
    tensor::Tensor t(DataType::kDataTypeFp32, 4, 4, true, alloc);
    ASSERT_EQ(t.is_empty(), false);
    t.index<float>(3) = 3.0;
    float* p = t.ptr<float>();
    ASSERT_EQ(p[3], 3.0);
}