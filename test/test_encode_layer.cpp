#include "op/encode.h"
#include <gtest/gtest.h>
TEST(test_encode_cpu, encode1){

    std::string token_model_path = "/home/zs/project/llama2/KuiperLLama/fushenshen/lession_model/tokenizer.model";
    op::SpeEncodeLayer speEncodeLayer(token_model_path, true, true);

    std::string s = "你好";
    int32_t token_id = 12;
    std::vector<int32_t> token_ids{11, 12, 13};
    std::vector<int32_t> ans;
    ans = speEncodeLayer.encode(s);
    for(auto i : ans){
        std::cout<<i<<" ";
    }
    std::cout<<speEncodeLayer.decode(token_ids)<<std::endl;
    std::cout<<speEncodeLayer.decode(token_id)<<std::endl;
    ASSERT_EQ(1,1);
}