#include "gpu_scheduler.h"
#include <string.h>

static uint32_t
buffers_overlap(const void *first, size_t first_size, const void *second, size_t second_size)
{
    return (uintptr_t)first < (uintptr_t)second + second_size
           && (uintptr_t)second < (uintptr_t)first + first_size;
}

error_t
gpu_scheduler_init(gpu_scheduler_t *scheduler, gpu_queue_t *const *queues, uint32_t count)
{
    if (NULL == scheduler || NULL == queues || 0 == count || count > GPU_SCHEDULER_QUEUE_LIMIT)
    {
        return POUND_ERROR_INVALID_ARGUMENT;
    }
    gpu_queue_t *const *cursor = queues;
    for (uint32_t index = 0; index < count; ++index, ++cursor)
    {
        if (NULL == *cursor || NULL == (*cursor)->channel)
        {
            return POUND_ERROR_INVALID_ARGUMENT;
        }
        const gpu_channel_t *channel = (*cursor)->channel;
        if (NULL == channel->command_buffer || NULL == channel->transfer_buffer
            || 0 == channel->command_capacity || 0 == channel->transfer_capacity
            || channel->command_capacity > UINTPTR_MAX - (uintptr_t)channel->command_buffer
            || channel->transfer_capacity > UINTPTR_MAX - (uintptr_t)channel->transfer_buffer)
        {
            return POUND_ERROR_INVALID_ARGUMENT;
        }
        uint32_t      completed = 0;
        const error_t status    = gpu_queue_drain(*cursor, (*cursor)->get, 0, &completed);
        if (POUND_SUCCESS != status)
        {
            return status;
        }
        gpu_queue_t *const *previous = queues;
        for (uint32_t other = 0; other < index; ++other, ++previous)
        {
            if (*previous == *cursor || (*previous)->channel == (*cursor)->channel)
            {
                return POUND_ERROR_INVALID_ARGUMENT;
            }
            const gpu_channel_t *other_channel = (*previous)->channel;
            if (0
                    != buffers_overlap(channel->command_buffer,
                                       channel->command_capacity,
                                       other_channel->command_buffer,
                                       other_channel->command_capacity)
                || 0
                       != buffers_overlap(channel->command_buffer,
                                          channel->command_capacity,
                                          other_channel->transfer_buffer,
                                          other_channel->transfer_capacity)
                || 0
                       != buffers_overlap(channel->transfer_buffer,
                                          channel->transfer_capacity,
                                          other_channel->command_buffer,
                                          other_channel->command_capacity))
            {
                return POUND_ERROR_INVALID_ARGUMENT;
            }
        }
    }
    memset(scheduler, 0, sizeof(*scheduler));
    gpu_scheduler_slot_t *slot = scheduler->slots;
    cursor                     = queues;
    for (uint32_t index = 0; index < count; ++index, ++slot, ++cursor)
    {
        slot->queue = *cursor;
        slot->put   = (*cursor)->get;
    }
    scheduler->count = count;
    return POUND_SUCCESS;
}

error_t
gpu_scheduler_publish(gpu_scheduler_t *scheduler, uint32_t slot, uint32_t put)
{
    if (NULL == scheduler || 0 == scheduler->count || scheduler->count > GPU_SCHEDULER_QUEUE_LIMIT
        || slot >= scheduler->count || NULL == scheduler->slots[slot].queue)
    {
        return POUND_ERROR_INVALID_ARGUMENT;
    }
    gpu_scheduler_slot_t *target    = &scheduler->slots[slot];
    uint32_t              completed = 0;
    const error_t         status    = gpu_queue_drain(target->queue, put, 0, &completed);
    if (POUND_SUCCESS == status)
    {
        target->put = put;
    }
    return status;
}

error_t
gpu_scheduler_run(gpu_scheduler_t *scheduler, uint32_t budget, gpu_scheduler_result_t *result)
{
    if (NULL == result)
    {
        return POUND_ERROR_INVALID_ARGUMENT;
    }
    *result = (gpu_scheduler_result_t) { .fault_slot = UINT32_MAX };
    if (NULL == scheduler || 0 == scheduler->count || scheduler->count > GPU_SCHEDULER_QUEUE_LIMIT
        || scheduler->next >= scheduler->count)
    {
        return POUND_ERROR_INVALID_ARGUMENT;
    }
    gpu_scheduler_slot_t *cursor = scheduler->slots;
    for (uint32_t index = 0; index < scheduler->count; ++index, ++cursor)
    {
        if (NULL == cursor->queue || NULL == cursor->queue->channel)
        {
            return POUND_ERROR_INVALID_ARGUMENT;
        }
    }
    uint32_t next            = scheduler->next;
    uint32_t visited         = 0;
    uint32_t progressed      = 0;
    uint32_t halted          = 0;
    uint32_t attempted       = 0;
    uint32_t completed_total = 0;
    uint32_t stalled         = 0;
    while (attempted < budget)
    {
        gpu_scheduler_slot_t *slot  = &scheduler->slots[next];
        gpu_queue_t          *queue = slot->queue;
        const uint32_t        bit   = UINT32_C(1) << next;
        if (0 == (halted & bit))
        {
            uint32_t       completed = 0;
            const uint64_t methods   = queue->channel->methods_executed;
            const uint32_t has_work  = queue->get != slot->put || 0 != queue->pending
                                      || 0 != queue->channel->pending_size;
            slot->status = gpu_queue_drain(queue, slot->put, 0 != has_work ? 1U : 0, &completed);
            if (0 != has_work || POUND_SUCCESS != slot->status)
            {
                ++attempted;
            }
            completed_total += completed;
            if (0 != completed || methods != queue->channel->methods_executed)
            {
                progressed = 1U;
            }
            if (POUND_SUCCESS != slot->status && POUND_ERROR_GPU_WAIT != slot->status
                && POUND_ERROR_GPU_BUSY != slot->status)
            {
                halted |= bit;
            }
        }
        next = next + 1U == scheduler->count ? 0 : next + 1U;
        ++visited;
        if (visited == scheduler->count)
        {
            if (0 == progressed)
            {
                stalled = 1U;
                break;
            }
            visited    = 0;
            progressed = 0;
        }
    }
    scheduler->next = next;
    gpu_scheduler_result_t summary
        = { .attempted = attempted, .completed = completed_total, .fault_slot = UINT32_MAX };
    error_t status = POUND_SUCCESS;
    cursor         = scheduler->slots;
    for (uint32_t index = 0; index < scheduler->count; ++index, ++cursor)
    {
        if (cursor->queue->get != cursor->put || 0 != cursor->queue->pending
            || POUND_SUCCESS != cursor->status)
        {
            ++summary.pending;
        }
        if (POUND_ERROR_GPU_WAIT == cursor->status)
        {
            ++summary.waiting;
        }
        else if (POUND_ERROR_GPU_BUSY == cursor->status)
        {
            ++summary.busy;
        }
        else if (POUND_SUCCESS != cursor->status)
        {
            ++summary.faulted;
            if (UINT32_MAX == summary.fault_slot)
            {
                summary.fault_slot = index;
                status             = cursor->status;
            }
        }
    }
    summary.budget_exhausted = attempted == budget && 0 != summary.pending && 0 == stalled;
    if (POUND_SUCCESS == status && 0 == summary.budget_exhausted)
    {
        status = 0 != summary.busy ? POUND_ERROR_GPU_BUSY
                                   : (0 != summary.waiting ? POUND_ERROR_GPU_WAIT : POUND_SUCCESS);
    }
    *result = summary;
    return status;
}
