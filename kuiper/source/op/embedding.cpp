#include "op/embedding.h"
#include "kernels/kernels_interface.h"
namespace op{

EmbeddingLayer::EmbeddingLayer(base::DeviceType device_type, int32_t dim, int32_t seq_len,
                                int32_t vocab_size)
    :dim_(dim),
    seq_len_(seq_len),
    vocab_size_(vocab_size),
    LayerParam(device_type, LayerType::kLayerEmbedding, false, "Embedding"){
        reset_weight_size(1);
        reset_input_size(1);
        reset_output_size(1);
}

base::Status EmbeddingLayer::check() const{
    return base::error::Success();
}

base::Status EmbeddingLayer::forward(){
    base::Status status = check();
    if(!status) return status;
    kernel::get_emb_kernel(device_type_)(get_input(0), get_weight(0), get_output(0), vocab_size_);

    return base::error::Success();
}

} // namespace op