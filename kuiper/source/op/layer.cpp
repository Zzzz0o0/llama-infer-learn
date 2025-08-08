#include "op/layer.h"

namespace op{
BaseLayer::BaseLayer(base::DeviceType device_type, LayerType layer_type, base::DataType data_type, std::string layer_name)
    :device_type_(device_type),
     layer_type_(layer_type),
     data_type_(data_type),
     layer_name_(std::move(layer_name)){}

base::Status BaseLayer::set_weight(int32_t idx, const tensor::Tensor& weight) {
  return base::error::FunctionNotImplement();
}

base::Status BaseLayer::set_weight(int32_t idx, const std::vector<int32_t>& dims,
                                   const void* weight_ptr, base::DeviceType device_type) {
  return base::error::FunctionNotImplement();
}

Layer::Layer(base::DeviceType device_type, LayerType layer_type, std::string layer_name)
    :BaseLayer(device_type, layer_type, base::DataType::kDataTypeFp32, std::move(layer_name)){}

base::Status Layer::init() { return base::error::Success(); }

base::Status Layer::check() const {
    return base::error::FunctionNotImplement("The check function is not implement yet");
}

base::Status Layer::forward() { return base::error::FunctionNotImplement(""); }

base::Status Layer::forward(const tensor::Tensor& input1, const tensor::Tensor& output1) {
    this->set_input(0, input1);
    this->set_output(0, output1);
    return this->forward();
}

void Layer::set_input(int32_t idx, const tensor::Tensor& input){
    this->inputs_.at(idx) = input;
}

void Layer::set_output(int32_t idx, const tensor::Tensor& output){
    this->outputs_.at(idx) = output;
}

tensor::Tensor& Layer::get_input(int32_t idx) {
    return this->inputs_.at(idx);
}

tensor::Tensor& Layer::get_output(int32_t idx) {
    return this->outputs_.at(idx);
}

const tensor::Tensor& Layer::get_input(int32_t idx) const {
  return inputs_.at(idx);
}

const tensor::Tensor& Layer::get_output(int32_t idx) const {
  return outputs_.at(idx);
}

void Layer::reset_input_size(size_t size) { inputs_.resize(size); }

void Layer::reset_output_size(size_t size) { outputs_.resize(size); }

LayerParam::LayerParam(base::DeviceType device_type, LayerType layer_type, bool is_quant_layer, std::string layer_name)
    :Layer(device_type, layer_type, std::move(layer_name)),is_quant_layer_(is_quant_layer){}

base::Status LayerParam::set_weight(int32_t idx, const tensor::Tensor& weight){
    weights_.at(idx) = weight;
    return base::error::Success();
}

base::Status LayerParam::set_weight(int32_t idx, const std::vector<int32_t>& dims,
                                    const void* weight_ptr, base::DeviceType device_type){
    // TODO
    return base::error::Success();               
}

tensor::Tensor& LayerParam::get_weight(int32_t idx){
    return this->weights_.at(idx);
}

const tensor::Tensor& LayerParam::get_weight(int32_t idx) const {
  return weights_.at(idx);
}

void LayerParam::set_scales(const tensor::Tensor& scales) {
  this->scales_ = scales;
}

} // namespace op