#ifndef POUND_GPU_QUEUE_H
#define POUND_GPU_QUEUE_H

#include "gpu_channel.h"

typedef struct
{
    gpu_channel_t *channel;
    uint64_t       address;
    uint32_t       capacity;
    uint32_t       get;
    uint32_t       pending;
} gpu_queue_t;

error_t gpu_queue_init(
    gpu_queue_t *queue, gpu_channel_t *channel, uint64_t address, uint32_t capacity, uint32_t get);
error_t gpu_queue_drain(gpu_queue_t *queue, uint32_t put, uint32_t budget, uint32_t *completed);

#endif
