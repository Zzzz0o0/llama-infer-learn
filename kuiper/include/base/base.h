#ifndef KUIPER_INCLUDE_BASE_BASE_H_
#define KUIPER_INCLUDE_BASE_BASE_H_

namespace base{
    enum class MemcpyKind{
        kMemcpyCPU2CPU = 0,
        kMemcpyCPU2CUDA = 1,
        kMemcpyCUDA2CPU = 2,
        kMemcpyCUDA2CUDA = 3,
    };

    enum class DeviceType : uint8_t{
        kDeviceUnknown = 0,
        kDeviceCPU = 1,
        kDeviceCUDA = 2,
    };

    enum class DataType : uint8_t{
        kDataTypeUnknown = 0,
        kDataTypeFp32 = 1,
        kDataTypeInt8 = 2,
        kDataTypeInt32 = 3,
    };

    inline size_t DataTypeSize(DataType data_type){
        if(data_type == DataType::kDataTypeFp32) return sizeof(float);
        else if(data_type == DataType::kDataTypeInt8) return sizeof(int8_t);
        else if(data_type == DataType::kDataTypeInt32) return sizeof(int32_t);
        else return 0;
    }

} // namespace base

#endif // KUIPER_INCLUDE_BASE_BASE_H_