#include <gtest/gtest.h>
#include "base/buffer.h"

TEST(test_buffer, allocate){
    using namespace base;
    auto alloc = base::CPUDeviceAllocatorFactory::get_instance();
    Buffer buffer(32, alloc);
    std::cout << "alloc use count: " << alloc.use_count() << std::endl;
    ASSERT_NE(buffer.ptr(), nullptr);
}