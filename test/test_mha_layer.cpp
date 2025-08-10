#include <gtest/gtest.h>
#include "base/base.h"
#include "tensor/tensor.h"
#include "op/mha.h"
TEST(test_mha_cpu, mha1){
    auto alloc = base::CPUDeviceAllocatorFactory::get_instance();
    int32_t seq_len = 3, pos = 0, head_num = 2, layer_index = 0, kv_dim = 4, kv_mul = 2, head_size = 4;
    tensor::Tensor q(base::DataType::kDataTypeFp32, head_num * head_size, true, alloc);
    tensor::Tensor k(base::DataType::kDataTypeFp32, seq_len, kv_dim, true, alloc);
    tensor::Tensor v(base::DataType::kDataTypeFp32, seq_len, kv_dim, true, alloc);
    tensor::Tensor score(base::DataType::kDataTypeFp32, head_num * seq_len, true, alloc);
    tensor::Tensor out(base::DataType::kDataTypeFp32, head_num * head_size, true, alloc);


    for(int i=0;i<head_num * head_size;++i){
        q.index<float>(i) = i * 1.0;
    }
    for(int i=0;i<kv_dim;++i){
        k.index<float>(i) = i * 1.0;
        v.index<float>(i) = i * 2.0;
    }
    
    op::MultiHeadAttention mha_layer(base::DeviceType::kDeviceCPU, layer_index, kv_mul, kv_dim, seq_len, head_num, head_size);
    mha_layer.set_output(0, out);
    mha_layer.set_input(0, q);
    mha_layer.set_input(1, score);
    mha_layer.set_input(2, k);
    mha_layer.set_input(3, v);
    mha_layer.forward();
    for(int i=0;i<head_num * head_size;++i){
        std::cout<<out.index<float>(i)<<" ";
    }
    std::cout<<std::endl;


    ASSERT_EQ(0, 0);
    
}