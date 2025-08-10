#include <gtest/gtest.h>
#include "base/base.h"
#include "tensor/tensor.h"
#include "op/embedding.h"
TEST(test_emb_cpu, emb1){
    auto alloc = base::CPUDeviceAllocatorFactory::get_instance();
    int32_t seq_len = 4, vocab_size = 6, embedding_dim = 3;
    tensor::Tensor input(base::DataType::kDataTypeFp32, seq_len, true, alloc);
    tensor::Tensor weight(base::DataType::kDataTypeFp32, vocab_size, embedding_dim, true, alloc);
    tensor::Tensor out(base::DataType::kDataTypeFp32, seq_len, embedding_dim, true, alloc);

    for(int i=0;i<seq_len;++i){
        input.index<float>(i) = seq_len-1-i;
    }
    for(int i =0;i<vocab_size;++i){
        for(int j=0;j<embedding_dim;j++){
            weight.index<float>(i*embedding_dim+j) = i*embedding_dim+j;
        }
    }

    op::EmbeddingLayer emb_layer(base::DeviceType::kDeviceCPU, embedding_dim, seq_len, vocab_size);
    emb_layer.set_input(0, input);
    emb_layer.set_weight(0, weight);
    emb_layer.set_output(0, out);
    emb_layer.forward();

    for(int i=0;i<seq_len;++i){
        for(int j=0;j<embedding_dim;++j){
            float t = out.index<float>(i*embedding_dim+j);
            std::cout<<t<<" ";
        }
        std::cout<<std::endl;
    }

    ASSERT_EQ(9.0, out.index<float>(0));
}