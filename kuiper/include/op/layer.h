#ifndef KUIPER_INCLUDE_OP_LAYER_H_
#define KUIPER_INCLUDE_OP_LAYER_H_
#include "base/base.h"
#include "tensor/tensor.h"
namespace op{
enum class LayerType : uint8_t{
    kLayerUnknown = 0,
    kLayerLinear = 1,
    kLayerAdd = 2,
};


class Layer;

class BaseLayer{
public:
    explicit BaseLayer(base::DeviceType device_type, LayerType layer_type, base::DataType data_type,
                        std::string layer_name="");
    
    virtual base::Status init() = 0; // 初始化算子

    virtual base::Status forward() = 0; // 前向推理

    virtual base::Status forward(const tensor::Tensor& input1, const tensor::Tensor& output1) = 0;

    virtual base::Status  check() const = 0; // 检查当前layer的输入输出权重大小是否符合

    virtual base::Status set_weight(int32_t idx, const tensor::Tensor& weight);

    virtual base::Status set_weight(int32_t dix, const std::vector<int32_t>& dims, const void* weight_ptr,
                                    base::DeviceType device_type = base::DeviceType::kDeviceUnknown);
    
    virtual void set_input(int32_t idx, const tensor::Tensor& input) = 0;

    virtual void set_output(int32_t idx, const tensor::Tensor& output) = 0;

    virtual tensor::Tensor& get_input(int32_t idx) = 0;

    virtual tensor::Tensor& get_output(int32_t idx) = 0;

    virtual const tensor::Tensor& get_input(int32_t idx) const = 0;

    virtual const tensor::Tensor& get_output(int32_t idx) const = 0;

protected:
    std::string layer_name_;
    LayerType layer_type_ = LayerType::kLayerUnknown;
    base::DataType data_type_ = base::DataType::kDataTypeUnknown;
    base::DeviceType device_type_ = base::DeviceType::kDeviceUnknown;
};

class Layer : public BaseLayer{
public:
    explicit Layer(base::DeviceType, LayerType layer_type, std::string layer_name="");

    base::Status init() override;

    base::Status check() const override;

    base::Status forward() override;

    base::Status forward(const tensor::Tensor& input1, const tensor::Tensor& output1) override;

    void set_input(int32_t idx, const tensor::Tensor& input) override;

    void set_output(int32_t idx, const tensor::Tensor& output) override;

    tensor::Tensor& get_input(int32_t idx) override;

    tensor::Tensor& get_output(int32_t idx) override;

    const tensor::Tensor& get_input(int32_t idx) const override;

    const tensor::Tensor& get_output(int32_t idx) const override;

    void reset_input_size(size_t size);

    void reset_output_size(size_t size);
    
protected:
    std::vector<tensor::Tensor> inputs_;
    std::vector<tensor::Tensor> outputs_;

};

class LayerParam : public Layer{
public:
    explicit LayerParam(base::DeviceType device_type, LayerType layer_type, bool is_quant_layer=false, std::string layer_name="");

    base::Status set_weight(int32_t idx, const tensor::Tensor& weight) override;

    base::Status set_weight(int32_t idx, const std::vector<int32_t>& dims, const void* weight_ptr,
                            base::DeviceType device_type = base::DeviceType::kDeviceUnknown) override;
    
    tensor::Tensor& get_weight(int32_t idx);

    const tensor::Tensor& get_weight(int32_t idx) const;

    void set_scales(const tensor::Tensor& scales);

protected:
    bool is_quant_layer_ = false;
    tensor::Tensor scales_;
    std::vector<tensor::Tensor> weights_;
};

} // namespace op

#endif // KUIPER_INCLUDE_OP_LAYER_H_