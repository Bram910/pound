#include "gpu_queue.h"

error_t
gpu_queue_init(
    gpu_queue_t *queue, gpu_channel_t *channel, uint64_t address, uint32_t capacity, uint32_t get)
{
    if (NULL == queue || NULL == channel || NULL == channel->memory || capacity < 2U
        || 0 != (capacity & (capacity - 1U)) || get >= capacity || 0 != (address & 7U))
    {
        return POUND_ERROR_INVALID_ARGUMENT;
    }
    const uint64_t limit = UINT64_C(1) << 40U;
    if (address >= limit || (uint64_t)capacity * 8U > limit - address)
    {
        return POUND_ERROR_GUEST_ADDRESS_OVERFLOW;
    }
    if (0 != channel->pending_size)
    {
        return POUND_ERROR_GPU_BUSY;
    }
    *queue = (gpu_queue_t) {
        .channel = channel, .address = address, .capacity = capacity, .get = get
    };
    return POUND_SUCCESS;
}

error_t
gpu_queue_drain(gpu_queue_t *queue, uint32_t put, uint32_t budget, uint32_t *completed)
{
    if (NULL == completed)
    {
        return POUND_ERROR_INVALID_ARGUMENT;
    }
    *completed = 0;
    if (NULL == queue || NULL == queue->channel || NULL == queue->channel->memory
        || queue->capacity < 2U || 0 != (queue->capacity & (queue->capacity - 1U))
        || put >= queue->capacity || queue->get >= queue->capacity)
    {
        return POUND_ERROR_INVALID_ARGUMENT;
    }
    if (0 == queue->pending && 0 != queue->channel->pending_size)
    {
        return POUND_ERROR_GPU_BUSY;
    }
    if (0 != queue->pending && (0 == queue->channel->pending_size || put == queue->get))
    {
        return POUND_ERROR_INVALID_ARGUMENT;
    }
    uint32_t get     = queue->get;
    uint32_t pending = queue->pending;
    uint32_t count   = 0;
    error_t  status  = POUND_SUCCESS;
    while (get != put && count < budget)
    {
        if (0 != pending)
        {
            status = gpu_channel_resume(queue->channel);
        }
        else
        {
            uint8_t entry[8];
            status = gpu_memory_read(
                queue->channel->memory, queue->address + (uint64_t)get * 8U, entry, sizeof(entry));
            if (POUND_SUCCESS == status)
            {
                status = gpu_channel_submit(queue->channel, entry, sizeof(entry));
            }
        }
        if (POUND_SUCCESS != status)
        {
            pending = 0 != queue->channel->pending_size;
            break;
        }
        pending = 0;
        get     = (get + 1U) & (queue->capacity - 1U);
        ++count;
    }
    queue->get     = get;
    queue->pending = pending;
    *completed     = count;
    return status;
}
