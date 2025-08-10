#include "tensor/tensor.h"

namespace tensor{
Tensor::Tensor(base::DataType data_type, int32_t dim0, bool need_alloc,
                std::shared_ptr<base::DeviceAllocator> alloc, void* ptr)
    :data_type_(data_type){
        dims_.push_back(dim0);
        size_ = dim0;
        if(need_alloc && alloc){
            allocate(alloc);
        }else{
            if(ptr!=nullptr){
                init_buffer(alloc, data_type_, need_alloc, ptr);
            }
        }
    }

Tensor::Tensor(base::DataType data_type, int32_t dim0, int32_t dim1, bool need_alloc,
                std::shared_ptr<base::DeviceAllocator> alloc, void* ptr)
    :data_type_(data_type){
        dims_.push_back(dim0);
        dims_.push_back(dim1);
        size_ = dim0 * dim1;
        if(alloc && need_alloc){
            allocate(alloc);
        }else{
            if(ptr!=nullptr){
                init_buffer(alloc, data_type_, need_alloc, ptr);
            }
        }
    }

Tensor::Tensor(base::DataType data_type, int32_t dim0, int32_t dim1, int32_t dim2, bool need_alloc,
               std::shared_ptr<base::DeviceAllocator> alloc, void* ptr)
    : data_type_(data_type) {
  dims_.push_back(dim0);
  dims_.push_back(dim1);
  dims_.push_back(dim2);
  size_ = dim0 * dim1 * dim2;
  if (need_alloc && alloc) {
    allocate(alloc);
  } else {
    init_buffer(alloc, data_type_, need_alloc, ptr);
  }
}

Tensor::Tensor(base::DataType data_type, int32_t dim0, int32_t dim1, int32_t dim2, int32_t dim3,
               bool need_alloc, std::shared_ptr<base::DeviceAllocator> alloc, void* ptr)
    : data_type_(data_type) {
  dims_.push_back(dim0);
  dims_.push_back(dim1);
  dims_.push_back(dim2);
  dims_.push_back(dim3);
  size_ = dim0 * dim1 * dim2 * dim3;
  if (need_alloc && alloc) {
    allocate(alloc);
  } else {
    init_buffer(alloc, data_type_, need_alloc, ptr);
  }
}

Tensor::Tensor(base::DataType data_type, std::vector<int32_t> dims, bool need_alloc,
                std::shared_ptr<base::DeviceAllocator> alloc, void* ptr)
    :data_type_(data_type), dims_(std::move(dims)){
        for(int i=0;i<dims_.size();i++){
            if(i==0) size_=dims[0];
            else size_*=dims[i];
        }
        if(need_alloc && alloc){
            allocate(alloc);
        }else{
            init_buffer(alloc, data_type_, need_alloc, ptr);
        }
    }


size_t Tensor::size() const { return this->size_; }

size_t Tensor::byte_size() const { return size_ * DataTypeSize(data_type_); }

bool Tensor::allocate(std::shared_ptr<base::DeviceAllocator> allocator, bool need_realloc){
    if(!allocator) return false;

    size_t byte_size = this->byte_size();
    
    if(!byte_size) return false;

    if(buffer_ && byte_size <= buffer_->byte_size()){
        if(~need_realloc){
            return true;
        }
    }

    buffer_ = std::make_shared<base::Buffer>(byte_size, allocator, nullptr);
    if(!buffer_->ptr()) return false;
    return true;
}

void Tensor::init_buffer(std::shared_ptr<base::DeviceAllocator> alloc, base::DataType data_type,
                        bool need_alloc, void* ptr){
    if(!alloc && !need_alloc){
        std::shared_ptr<base::Buffer> buffer = std::make_shared<base::Buffer>(this->byte_size(), nullptr, ptr, true);
        this->buffer_ = buffer;
    }else{
        allocate(alloc, true);
    }
}

base::DataType Tensor::data_type() const{
    return this->data_type_;
}

int32_t Tensor::get_dim(int32_t idx) const{
    return this->dims_.at(idx);
}

const std::vector<int32_t>& Tensor::dims() const{
    return this->dims_;
}

bool Tensor::is_empty() const{
    if(size_!=0) return false;
    else return true;
}

void Tensor::set_device_type(base::DeviceType device_type) const {
  if (buffer_) {
    buffer_->set_device_type(device_type);
  }
}

} //namespace tensor