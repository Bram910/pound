#include "gpu_semaphore.h"

error_t
gpu_semaphore_execute_timed(const gpu_memory_t *memory,
                            uint64_t            address,
                            uint64_t            payload,
                            uint32_t            flags,
                            gpu_clock_read_t    read_clock,
                            void               *clock_context)
{
    if (NULL == memory)
    {
        return POUND_ERROR_INVALID_ARGUMENT;
    }
    const uint32_t operation   = flags & 7U;
    const uint32_t reduction   = (flags >> 27U) & 15U;
    const uint32_t is_unsigned = flags >> 31U;
    const size_t   size        = 0 != (flags & 0x01000000U) ? 8U : 4U;
    const uint32_t timestamp   = 0 != (flags & 0x02000000U);
    uint32_t       allowed     = 0x01000007U;
    if (1U == operation || 6U == operation)
    {
        allowed |= 0x02100000U;
    }
    if (6U == operation)
    {
        allowed |= 0xF8000000U;
    }
    if (operation > 6U || 0 != (flags & ~allowed)
        || (6U == operation
            && (reduction > 7U || (reduction >= 6U && (8U == size || 0 == is_unsigned))
                || (5U == reduction && 8U == size && 0 == is_unsigned))))
    {
        return POUND_ERROR_GPU_UNSUPPORTED_COMMAND;
    }
    if (0 != address % (0 != timestamp ? 16U : size))
    {
        return POUND_ERROR_GPU_INVALID_COMMAND;
    }
    if (0 != timestamp)
    {
        if (NULL == read_clock)
        {
            return POUND_ERROR_GPU_UNSUPPORTED_COMMAND;
        }
        const error_t status = gpu_memory_validate(memory, address, 16U, GPU_MEMORY_WRITE);
        if (POUND_SUCCESS != status)
        {
            return status;
        }
    }
    const uint64_t mask      = 8U == size ? UINT64_MAX : UINT32_MAX;
    const uint64_t sign      = 8U == size ? UINT64_C(1) << 63U : UINT64_C(1) << 31U;
    const uint64_t expected  = payload & mask;
    uint8_t        bytes[16] = { 0 };
    uint64_t       value     = expected;
    if (1U != operation)
    {
        const error_t status = gpu_memory_read(memory, address, bytes, size);
        if (POUND_SUCCESS != status)
        {
            return status;
        }
        value = 0;
        for (size_t index = 0; index < size; ++index)
        {
            value |= (uint64_t)bytes[index] << (index * 8U);
        }
    }
    uint32_t satisfied = 0;
    switch (operation)
    {
        case 0U:
            satisfied = value == expected;
            break;
        case 2U:
            satisfied = value >= expected;
            break;
        case 3U:
            satisfied = ((value - expected) & mask) < sign;
            break;
        case 4U:
            satisfied = 0 != (value & expected);
            break;
        case 5U:
            satisfied = 0 != (~(value | expected) & mask);
            break;
        case 6U: {
            const uint64_t bias = 0 != is_unsigned ? 0 : sign;
            switch (reduction)
            {
                case 0U:
                    value = (value ^ bias) < (expected ^ bias) ? value : expected;
                    break;
                case 1U:
                    value = (value ^ bias) > (expected ^ bias) ? value : expected;
                    break;
                case 2U:
                    value ^= expected;
                    break;
                case 3U:
                    value &= expected;
                    break;
                case 4U:
                    value |= expected;
                    break;
                case 5U:
                    value += expected;
                    break;
                case 6U:
                    value = value >= expected ? 0 : value + 1U;
                    break;
                case 7U:
                    value = 0 == value || value > expected ? expected : value - 1U;
                    break;
                default:
                    return POUND_ERROR_GPU_UNSUPPORTED_COMMAND;
            }
            break;
        }
        default:
            break;
    }
    if (1U != operation && 6U != operation)
    {
        return 0 != satisfied ? POUND_SUCCESS : POUND_ERROR_GPU_WAIT;
    }
    for (size_t index = 0; index < size; ++index)
    {
        bytes[index] = (uint8_t)(value >> (index * 8U));
    }
    if (0 != timestamp)
    {
        const uint64_t time = read_clock(clock_context);
        for (size_t index = 0; index < 8U; ++index)
        {
            bytes[8U + index] = (uint8_t)(time >> (index * 8U));
        }
        const error_t status = gpu_memory_write(memory, address + 8U, bytes + 8U, 8U);
        return POUND_SUCCESS == status ? gpu_memory_write(memory, address, bytes, 8U) : status;
    }
    return gpu_memory_write(memory, address, bytes, size);
}

error_t
gpu_semaphore_execute(const gpu_memory_t *memory,
                      uint64_t            address,
                      uint64_t            payload,
                      uint32_t            flags)
{
    return gpu_semaphore_execute_timed(memory, address, payload, flags, NULL, NULL);
}
