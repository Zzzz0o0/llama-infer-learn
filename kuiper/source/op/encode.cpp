#include "op/encode.h"
#include<iostream>
namespace op{

SpeEncodeLayer::SpeEncodeLayer(std::string token_model_path, bool has_bos, bool has_eos)
    :EncodeLayerBase(std::move(token_model_path), has_bos, has_eos){
        using namespace sentencepiece::util;
        spe = std::make_unique<sentencepiece::SentencePieceProcessor>();
        auto rc = spe->Load(token_model_path_);
        if(rc.code()!= StatusCode::kOk){
            std::cout<<"导入失败"<<std::endl;
        }
}

std::vector<int32_t> SpeEncodeLayer::encode(const std::string& sentence) const{
    std::vector<int32_t> input_ids = spe->EncodeAsIds(sentence);
    if(has_bos_) input_ids.insert(input_ids.begin(), spe->bos_id());
    if(has_eos_) input_ids.push_back(spe->eos_id());
    return input_ids;
}

std::string SpeEncodeLayer::decode(int32_t token_id) const {
    std::vector<int32_t> token_ids{token_id};
    return this->spe->DecodeIds(token_ids);
}

std::string SpeEncodeLayer::decode(const std::vector<int32_t>& token_ids) const{
    return this->spe->DecodeIds(token_ids);
}

bool SpeEncodeLayer::is_sentence_ending(int32_t token_id) const{
    return token_id == this->spe->eos_id();
}

int32_t SpeEncodeLayer::vocab_size() const{
    return spe->GetPieceSize();
}

} // namespace op