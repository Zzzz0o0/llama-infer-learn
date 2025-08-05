#ifndef KUIPER_INCLUDE_BASE_BUFFER_H_
#define KUIPER_INCLUDE_BASE_BUFFER_H_
#include "alloc.h"
namespace base{
class Buffer{
private:
    size_t byte_size_ = 0;
    void* ptr_ = nullptr;
    bool use_external_ = false;
    DeviceType device_type_ = DeviceType::kDeviceUnknown;
    std::shared_ptr<DeviceAllocator> allocator_;
public:
    explicit Buffer() = default;
    explicit Buffer(size_t byte_size, std::shared_ptr<DeviceAllocator> allocator = nullptr,
                    void* ptr=nullptr, bool use_external = false);
    virtual ~Buffer();
    bool allocate();

    size_t byte_size() const;
    void* ptr();
    const void* ptr() const;
    bool is_external() const;
    DeviceType device_type() const;
    std::shared_ptr<DeviceAllocator> allocator() const;
};
} // namespace base
#endif // KUIPER_INCLUDE_BASE_BUFFER_H_