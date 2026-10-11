#include "gpu_channel.h"
#include "gpu_semaphore.h"
#include <string.h>

static uint32_t
has_active_upload(const gpu_channel_t *channel)
{
    const gpu_inline_t *cursor = channel->uploads;
    for (uint32_t index = 0; index < 8U; ++index, ++cursor)
    {
        if (0 != cursor->active)
        {
            return 1U;
        }
    }
    return 0;
}

static error_t
execute_method(gpu_channel_t *channel, uint32_t subchannel, uint32_t method, uint32_t data)
{
    switch (method)
    {
        case 0x00U:
            if (GPU_CLASS_AMPERE_DMA_COPY != data && GPU_CLASS_KEPLER_INLINE_TO_MEMORY != data)
            {
                return POUND_ERROR_GPU_UNSUPPORTED_COMMAND;
            }
            if (0 != channel->uploads[subchannel].active)
            {
                return POUND_ERROR_GPU_BUSY;
            }
            channel->classes[subchannel] = data;
            return POUND_SUCCESS;
        case 0x08U:
            return POUND_SUCCESS;
        case 0x28U:
        case 0x2CU:
        case 0x30U: {
            const uint32_t index              = (method - 0x28U) / 4U;
            channel->memory_operations[index] = data;
            channel->memory_operations_ready |= 1U << index;
            return POUND_SUCCESS;
        }
        case 0x34U:
            if (7U != channel->memory_operations_ready)
            {
                return POUND_ERROR_GPU_INVALID_COMMAND;
            }
            if (0x28000000U != data || 0 != channel->memory_operations[0]
                || 0 != channel->memory_operations[1] || channel->memory_operations[2] > 1U)
            {
                return POUND_ERROR_GPU_UNSUPPORTED_COMMAND;
            }
            channel->memory_operations_ready = 0;
            ++channel->barriers_executed;
            return POUND_SUCCESS;
        case 0x50U:
            if (0 != has_active_upload(channel))
            {
                return POUND_ERROR_GPU_BUSY;
            }
            channel->reference = data;
            return POUND_SUCCESS;
        case 0x5CU:
            if (0 != (data & 3U))
            {
                return POUND_ERROR_GPU_INVALID_COMMAND;
            }
            channel->semaphore_address
                = (channel->semaphore_address & UINT64_C(0xFF00000000)) | data;
            return POUND_SUCCESS;
        case 0x60U:
            if (data > 0xFFU)
            {
                return POUND_ERROR_GPU_INVALID_COMMAND;
            }
            channel->semaphore_address
                = (channel->semaphore_address & UINT64_C(0xFFFFFFFF)) | ((uint64_t)data << 32U);
            return POUND_SUCCESS;
        case 0x64U:
            channel->semaphore_payload
                = (channel->semaphore_payload & UINT64_C(0xFFFFFFFF00000000)) | data;
            return POUND_SUCCESS;
        case 0x68U:
            channel->semaphore_payload
                = (channel->semaphore_payload & UINT64_C(0xFFFFFFFF)) | ((uint64_t)data << 32U);
            return POUND_SUCCESS;
        case 0x6CU:
            if (0 != (data & 0x100000U) && (1U == (data & 7U) || 6U == (data & 7U))
                && 0 != has_active_upload(channel))
            {
                return POUND_ERROR_GPU_BUSY;
            }
            return gpu_semaphore_execute_timed(channel->memory,
                                               channel->semaphore_address,
                                               channel->semaphore_payload,
                                               data,
                                               channel->read_clock,
                                               channel->clock_context);
        case 0x78U:
            if (data > 1U)
            {
                return POUND_ERROR_GPU_INVALID_COMMAND;
            }
            return 0 == has_active_upload(channel) ? POUND_SUCCESS : POUND_ERROR_GPU_BUSY;
        default:
            break;
    }
    if (method >= 0x100U && GPU_CLASS_KEPLER_INLINE_TO_MEMORY == channel->classes[subchannel])
    {
        gpu_inline_t *upload = &channel->uploads[subchannel];
        const error_t status
            = gpu_inline_method(upload, channel->memory, method, data, channel->transfer_capacity);
        if (POUND_SUCCESS == status && 0 == upload->active
            && (0x1B0U == method || 0x1B4U == method))
        {
            ++channel->uploads_executed;
        }
        return status;
    }
    if (method < 0x100U || GPU_CLASS_AMPERE_DMA_COPY != channel->classes[subchannel])
    {
        return POUND_ERROR_GPU_UNSUPPORTED_COMMAND;
    }
    const error_t status = gpu_copy_method(&channel->copies[subchannel],
                                           channel->memory,
                                           method,
                                           data,
                                           channel->transfer_buffer,
                                           channel->transfer_capacity);
    if (POUND_SUCCESS == status && 0x300U == method && 0 != (data & 3U))
    {
        ++channel->copies_executed;
    }
    return status;
}

static error_t
dispatch_method(void *context, uint32_t subchannel, uint32_t method, uint32_t data)
{
    gpu_channel_t *channel = context;
    const error_t  status  = execute_method(channel, subchannel, method, data);
    if (POUND_SUCCESS == status)
    {
        ++channel->methods_executed;
    }
    return status;
}

error_t
gpu_channel_init(gpu_channel_t *channel,
                 gpu_memory_t  *memory,
                 void          *command_buffer,
                 size_t         command_capacity,
                 void          *transfer_buffer,
                 size_t         transfer_capacity)
{
    if (NULL == channel || NULL == memory || NULL == command_buffer || NULL == transfer_buffer
        || 0 == command_capacity || 0 != command_capacity % 4U || 0 == transfer_capacity
        || command_capacity > UINTPTR_MAX - (uintptr_t)command_buffer
        || transfer_capacity > UINTPTR_MAX - (uintptr_t)transfer_buffer)
    {
        return POUND_ERROR_INVALID_ARGUMENT;
    }
    const uintptr_t command_start  = (uintptr_t)command_buffer;
    const uintptr_t transfer_start = (uintptr_t)transfer_buffer;
    if (command_start < transfer_start + transfer_capacity
        && transfer_start < command_start + command_capacity)
    {
        return POUND_ERROR_INVALID_ARGUMENT;
    }

    memset(channel, 0, sizeof(*channel));
    channel->memory            = memory;
    channel->command_buffer    = command_buffer;
    channel->command_capacity  = command_capacity;
    channel->transfer_buffer   = transfer_buffer;
    channel->transfer_capacity = transfer_capacity;
    return gpu_pushbuffer_init(&channel->decoder);
}

error_t
gpu_channel_submit(gpu_channel_t *channel, const void *entry_bytes, size_t entry_size)
{
    if (NULL == channel || NULL == channel->memory || NULL == channel->command_buffer)
    {
        return POUND_ERROR_INVALID_ARGUMENT;
    }
    if (0 != channel->pending_size)
    {
        return POUND_ERROR_GPU_BUSY;
    }

    gpu_gpfifo_entry_t entry;
    error_t            status = gpu_gpfifo_decode(entry_bytes, entry_size, &entry);
    if (POUND_SUCCESS != status)
    {
        return status;
    }
    if (0 != (entry.flags & ~GPU_GPFIFO_SYNC_WAIT))
    {
        return POUND_ERROR_GPU_UNSUPPORTED_COMMAND;
    }
    const size_t size = (size_t)entry.word_count * 4U;
    if (size > channel->command_capacity)
    {
        return POUND_ERROR_GPU_RESOURCE_LIMIT;
    }
    status = gpu_memory_read(channel->memory, entry.address, channel->command_buffer, size);
    if (POUND_SUCCESS != status)
    {
        return status;
    }
    channel->pending_size   = size;
    channel->pending_offset = 0;
    return gpu_channel_resume(channel);
}

error_t
gpu_channel_resume(gpu_channel_t *channel)
{
    if (NULL == channel || NULL == channel->memory || NULL == channel->command_buffer
        || channel->pending_offset > channel->pending_size
        || channel->pending_size > channel->command_capacity)
    {
        return POUND_ERROR_INVALID_ARGUMENT;
    }
    if (0 == channel->pending_size)
    {
        return POUND_SUCCESS;
    }

    size_t        consumed = 0;
    const error_t status   = gpu_pushbuffer_decode(&channel->decoder,
                                                 channel->command_buffer + channel->pending_offset,
                                                 channel->pending_size - channel->pending_offset,
                                                 dispatch_method,
                                                 channel,
                                                 &consumed);
    channel->pending_offset += consumed;
    if (POUND_SUCCESS == status)
    {
        channel->pending_size   = 0;
        channel->pending_offset = 0;
    }
    return status;
}

error_t
gpu_channel_finish(const gpu_channel_t *channel)
{
    if (NULL == channel)
    {
        return POUND_ERROR_INVALID_ARGUMENT;
    }
    if (0 != channel->pending_size)
    {
        return POUND_ERROR_GPU_BUSY;
    }
    return 0 == has_active_upload(channel) ? gpu_pushbuffer_finish(&channel->decoder)
                                           : POUND_ERROR_GPU_TRUNCATED_COMMAND;
}

error_t
gpu_channel_reset(gpu_channel_t *channel)
{
    if (NULL == channel)
    {
        return POUND_ERROR_INVALID_ARGUMENT;
    }
    const gpu_sector_layout_t layout        = channel->sector_layout;
    const gpu_clock_read_t    read_clock    = channel->read_clock;
    void *const               clock_context = channel->clock_context;
    const error_t             status        = gpu_channel_init(channel,
                                            channel->memory,
                                            channel->command_buffer,
                                            channel->command_capacity,
                                            channel->transfer_buffer,
                                            channel->transfer_capacity);
    if (POUND_SUCCESS != status)
    {
        return status;
    }
    channel->read_clock    = read_clock;
    channel->clock_context = clock_context;
    return gpu_channel_set_sector_layout(channel, layout);
}

error_t
gpu_channel_set_clock(gpu_channel_t *channel, gpu_clock_read_t read_clock, void *clock_context)
{
    if (NULL == channel || NULL == channel->memory || (NULL == read_clock && NULL != clock_context))
    {
        return POUND_ERROR_INVALID_ARGUMENT;
    }
    if (0 != channel->pending_size || 0 != channel->decoder.remaining
        || 0 != has_active_upload(channel))
    {
        return POUND_ERROR_GPU_BUSY;
    }
    channel->read_clock    = read_clock;
    channel->clock_context = clock_context;
    return POUND_SUCCESS;
}

error_t
gpu_channel_set_sector_layout(gpu_channel_t *channel, gpu_sector_layout_t layout)
{
    if (NULL == channel || NULL == channel->memory
        || (GPU_SECTOR_LAYOUT_UNSPECIFIED != layout && GPU_SECTOR_LAYOUT_16BX2 != layout
            && GPU_SECTOR_LAYOUT_TURING2D != layout))
    {
        return POUND_ERROR_INVALID_ARGUMENT;
    }
    if (0 != channel->pending_size || 0 != channel->decoder.remaining
        || 0 != has_active_upload(channel))
    {
        return POUND_ERROR_GPU_BUSY;
    }
    gpu_copy_t   *cursor = channel->copies;
    gpu_inline_t *upload = channel->uploads;
    for (uint32_t index = 0; index < 8U; ++index)
    {
        cursor->sector_layout = layout;
        upload->sector_layout = layout;
        ++cursor;
        ++upload;
    }
    channel->sector_layout = layout;
    return POUND_SUCCESS;
}
