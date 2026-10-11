#ifndef POUND_GPU_SCHEDULER_H
#define POUND_GPU_SCHEDULER_H

#include "gpu_queue.h"

#define GPU_SCHEDULER_QUEUE_LIMIT 32U

typedef struct
{
    gpu_queue_t *queue;
    uint32_t     put;
    error_t      status;
} gpu_scheduler_slot_t;

typedef struct
{
    gpu_scheduler_slot_t slots[GPU_SCHEDULER_QUEUE_LIMIT];
    uint32_t             count;
    uint32_t             next;
} gpu_scheduler_t;

typedef struct
{
    uint32_t attempted;
    uint32_t completed;
    uint32_t pending;
    uint32_t waiting;
    uint32_t busy;
    uint32_t faulted;
    uint32_t fault_slot;
    uint32_t budget_exhausted;
} gpu_scheduler_result_t;

error_t gpu_scheduler_init(gpu_scheduler_t *scheduler, gpu_queue_t *const *queues, uint32_t count);
error_t gpu_scheduler_publish(gpu_scheduler_t *scheduler, uint32_t slot, uint32_t put);
error_t gpu_scheduler_run(gpu_scheduler_t        *scheduler,
                          uint32_t                budget,
                          gpu_scheduler_result_t *result);

#endif
