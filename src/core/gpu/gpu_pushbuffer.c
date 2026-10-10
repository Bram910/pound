#include "gpu_pushbuffer.h"
#include <string.h>

#define GPU_PACKET_INCREMENTING     1U
#define GPU_PACKET_NON_INCREMENTING 3U
#define GPU_PACKET_IMMEDIATE        4U
#define GPU_PACKET_INCREMENT_ONCE   5U
#define GPU_PACKET_END_SEGMENT      7U
#define GPU_METHOD_LIMIT            0xFFFU
#define GPU_PACKET_COUNT_LIMIT      0x1FFFU

static uint32_t
read_u32_le(const uint8_t *bytes)
{
    return (uint32_t)bytes[0] | ((uint32_t)bytes[1] << 8U) | ((uint32_t)bytes[2] << 16U)
           | ((uint32_t)bytes[3] << 24U);
}

error_t
gpu_pushbuffer_init(gpu_pushbuffer_t *decoder)
{
    if (NULL == decoder)
    {
        return POUND_ERROR_INVALID_ARGUMENT;
    }

    memset(decoder, 0, sizeof(*decoder));
    return POUND_SUCCESS;
}

error_t
gpu_pushbuffer_decode(gpu_pushbuffer_t    *decoder,
                      const void          *buffer,
                      size_t               byte_size,
                      gpu_method_handler_t handler,
                      void                *context,
                      size_t              *bytes_consumed)
{
    if (NULL == bytes_consumed)
    {
        return POUND_ERROR_INVALID_ARGUMENT;
    }

    *bytes_consumed = 0;
    if (NULL == decoder || NULL == handler || (NULL == buffer && 0 != byte_size)
        || 0 != byte_size % sizeof(uint32_t))
    {
        return POUND_ERROR_INVALID_ARGUMENT;
    }

    uint32_t method     = decoder->method;
    uint32_t remaining  = decoder->remaining;
    uint32_t subchannel = decoder->subchannel;
    uint32_t opcode     = decoder->opcode;

    if (method > GPU_METHOD_LIMIT || subchannel > 7U || remaining > GPU_PACKET_COUNT_LIMIT
        || (0 != remaining && GPU_PACKET_INCREMENTING != opcode
            && GPU_PACKET_NON_INCREMENTING != opcode && GPU_PACKET_INCREMENT_ONCE != opcode))
    {
        return POUND_ERROR_INVALID_ARGUMENT;
    }

    const uint8_t *cursor   = buffer;
    size_t         consumed = 0;
    error_t        status   = POUND_SUCCESS;
    decoder->segment_ended  = 0;

    while (consumed < byte_size)
    {
        const uint32_t word = read_u32_le(cursor);

        if (0 != remaining)
        {
            status = handler(context, subchannel, method * 4U, word);
            if (POUND_SUCCESS != status)
            {
                break;
            }

            --remaining;
            if (0 != remaining && GPU_PACKET_NON_INCREMENTING != opcode)
            {
                ++method;
                if (GPU_PACKET_INCREMENT_ONCE == opcode)
                {
                    opcode = GPU_PACKET_NON_INCREMENTING;
                }
            }
        }
        else if (0 != word)
        {
            const uint32_t next_opcode     = word >> 29U;
            const uint32_t next_method     = word & GPU_METHOD_LIMIT;
            const uint32_t count           = (word >> 16U) & GPU_PACKET_COUNT_LIMIT;
            const uint32_t next_subchannel = (word >> 13U) & 7U;

            if (GPU_PACKET_END_SEGMENT == next_opcode)
            {
                consumed               = byte_size;
                decoder->segment_ended = 1;
                break;
            }

            if (GPU_PACKET_INCREMENTING != next_opcode && GPU_PACKET_NON_INCREMENTING != next_opcode
                && GPU_PACKET_INCREMENT_ONCE != next_opcode && GPU_PACKET_IMMEDIATE != next_opcode)
            {
                status = POUND_ERROR_GPU_UNSUPPORTED_COMMAND;
                break;
            }

            if (0 == count && GPU_PACKET_IMMEDIATE != next_opcode)
            {
                cursor += sizeof(uint32_t);
                consumed += sizeof(uint32_t);
                continue;
            }

            const uint32_t increments = GPU_PACKET_INCREMENTING == next_opcode ? count - 1U
                                        : GPU_PACKET_INCREMENT_ONCE == next_opcode && count > 1U
                                            ? 1U
                                            : 0U;

            if (0 != (word & 0x1000U) || increments > GPU_METHOD_LIMIT - next_method)
            {
                status = POUND_ERROR_GPU_INVALID_COMMAND;
                break;
            }

            if (GPU_PACKET_IMMEDIATE == next_opcode)
            {
                status = handler(context, next_subchannel, next_method * 4U, count);
                if (POUND_SUCCESS != status)
                {
                    break;
                }
            }
            else
            {
                method     = next_method;
                subchannel = next_subchannel;
                remaining  = count;
                opcode     = next_opcode;
            }
        }

        cursor += sizeof(uint32_t);
        consumed += sizeof(uint32_t);
    }

    decoder->method     = method;
    decoder->remaining  = remaining;
    decoder->subchannel = subchannel;
    decoder->opcode     = opcode;
    *bytes_consumed     = consumed;
    return status;
}

error_t
gpu_pushbuffer_finish(const gpu_pushbuffer_t *decoder)
{
    if (NULL == decoder)
    {
        return POUND_ERROR_INVALID_ARGUMENT;
    }

    return 0 == decoder->remaining ? POUND_SUCCESS : POUND_ERROR_GPU_TRUNCATED_COMMAND;
}

error_t
gpu_gpfifo_decode(const void *buffer, size_t byte_size, gpu_gpfifo_entry_t *entry)
{
    if (NULL == buffer || NULL == entry || 8U != byte_size)
    {
        return POUND_ERROR_INVALID_ARGUMENT;
    }

    const uint8_t *bytes      = buffer;
    const uint32_t low        = read_u32_le(bytes);
    const uint32_t high       = read_u32_le(bytes + 4U);
    const uint32_t word_count = (high >> 10U) & 0x1FFFFFU;

    if (0 != (low & 2U) || 0 != (high & 0x100U))
    {
        return POUND_ERROR_GPU_INVALID_COMMAND;
    }

    if (0 == word_count)
    {
        if (0U != low || 0U != high)
        {
            return POUND_ERROR_GPU_UNSUPPORTED_COMMAND;
        }

        memset(entry, 0, sizeof(*entry));
        return POUND_SUCCESS;
    }

    entry->address    = ((uint64_t)(high & 0xFFU) << 32U) | (low & 0xFFFFFFFCU);
    entry->word_count = word_count;
    entry->flags      = (low & 1U) | ((high >> 8U) & GPU_GPFIFO_LEVEL_SUBROUTINE)
                   | ((high >> 29U) & GPU_GPFIFO_SYNC_WAIT);
    return POUND_SUCCESS;
}
