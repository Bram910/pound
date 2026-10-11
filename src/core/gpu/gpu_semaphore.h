#ifndef POUND_GPU_SEMAPHORE_H
#define POUND_GPU_SEMAPHORE_H

#include "gpu_memory.h"

typedef uint64_t (*gpu_clock_read_t)(void *context);

error_t gpu_semaphore_execute_timed(const gpu_memory_t *memory,
                                    uint64_t            address,
                                    uint64_t            payload,
                                    uint32_t            flags,
                                    gpu_clock_read_t    read_clock,
                                    void               *clock_context);
error_t gpu_semaphore_execute(const gpu_memory_t *memory,
                              uint64_t            address,
                              uint64_t            payload,
                              uint32_t            flags);

#endif
