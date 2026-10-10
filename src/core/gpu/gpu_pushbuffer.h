#ifndef POUND_GPU_PUSHBUFFER_H
#define POUND_GPU_PUSHBUFFER_H

#include "errors.h"
#include <stddef.h>
#include <stdint.h>

typedef struct
{
    uint32_t method;
    uint32_t remaining;
    uint32_t subchannel;
    uint32_t opcode;
    uint32_t segment_ended;
} gpu_pushbuffer_t;

typedef error_t (*gpu_method_handler_t)(void    *context,
                                        uint32_t subchannel,
                                        uint32_t method_address,
                                        uint32_t data);

typedef struct
{
    uint64_t address;
    uint32_t word_count;
    uint32_t flags;
} gpu_gpfifo_entry_t;

#define GPU_GPFIFO_FETCH_CONDITIONAL 1U
#define GPU_GPFIFO_LEVEL_SUBROUTINE  2U
#define GPU_GPFIFO_SYNC_WAIT         4U

error_t gpu_pushbuffer_init(gpu_pushbuffer_t *decoder);
error_t gpu_pushbuffer_decode(gpu_pushbuffer_t    *decoder,
                              const void          *buffer,
                              size_t               byte_size,
                              gpu_method_handler_t handler,
                              void                *context,
                              size_t              *bytes_consumed);
error_t gpu_pushbuffer_finish(const gpu_pushbuffer_t *decoder);
error_t gpu_gpfifo_decode(const void *buffer, size_t byte_size, gpu_gpfifo_entry_t *entry);

#endif
