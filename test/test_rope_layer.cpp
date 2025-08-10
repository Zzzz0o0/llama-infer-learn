#include <gtest/gtest.h>
#include "base/base.h"
#include "tensor/tensor.h"
#include "op/rope.h"
#include "../kuiper/source/op/kernels/cpu/rope_kernel.h"
TEST(test_rope_cpu, rope1){
    auto alloc = base::CPUDeviceAllocatorFactory::get_instance();
    int32_t seq_len = 2, dim = 8, kv_dim = 4, head_size = 4;
    tensor::Tensor input_q(base::DataType::kDataTypeFp32, dim, true, alloc);
    tensor::Tensor input_k(base::DataType::kDataTypeFp32, kv_dim, true, alloc);
    tensor::Tensor input_pos(base::DataType::kDataTypeInt32, 1, true, alloc);
    tensor::Tensor sin_cache(base::DataType::kDataTypeFp32, head_size, true, alloc);
    tensor::Tensor cos_cache(base::DataType::kDataTypeFp32, head_size, true, alloc);

    kernel::sin_cos_cache_calc_cpu(head_size, 1, sin_cache.ptr<float>(), cos_cache.ptr<float>());

    for(int i=0;i<dim;++i){
        input_q.index<float>(i) = i * 1.0;
    }
    for(int i=0;i<kv_dim;++i){
        input_k.index<float>(i) = i * 2.0;
    }
    
    // kernel::rope_kernel_cpu(dim, kv_dim, head_size, input_q, input_k, input_pos, sin_cache, cos_cache);


    op::RoPELayer rope_layer(base::DeviceType::kDeviceCPU, dim, kv_dim, head_size);
    rope_layer.set_input(0, input_q);
    rope_layer.set_input(1, input_k);
    rope_layer.set_input(2, input_pos);
    rope_layer.set_input(3, sin_cache);
    rope_layer.set_input(4, cos_cache);
    rope_layer.forward();
    for(int i=0;i<head_size;++i){
        std::cout<<sin_cache.index<float>(i)<<" ";
    }
    std::cout<<std::endl;
    for(int i=0;i<head_size;++i){
        std::cout<<cos_cache.index<float>(i)<<" ";
    }
    std::cout<<std::endl;
    for(int i=0;i<dim;++i){
        std::cout<<input_q.index<float>(i)<<" ";
    }
    std::cout<<std::endl;
    for(int i=0;i<kv_dim;++i){
        std::cout<<input_k.index<float>(i)<<" ";
    }
    std::cout<<std::endl;


    ASSERT_EQ(input_q.index<float>(0), 0);
    
}